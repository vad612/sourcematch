use anyhow::{Context, Result, anyhow, ensure};
use reqwest::{StatusCode, Url, blocking::Client, redirect::Policy};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path, sync::Mutex, thread, time::Duration};

const QUERY_INSTRUCTION: &str = "Instruct: Find the same product model in the supplier catalog, matching material, dimensions, series, and features.\nQuery: ";

#[derive(Deserialize)]
pub struct Profile {
    pub embedding: EmbeddingProfile,
    pub reranking: RerankingProfile,
    pub search: SearchProfile,
}

#[derive(Deserialize)]
pub struct EmbeddingProfile {
    pub url: String,
    pub model: String,
    pub dimension: usize,
    pub revision: String,
    pub batch_size: usize,
    #[serde(default)]
    pub check_loaded: bool,
}

#[derive(Deserialize)]
pub struct RerankingProfile {
    pub url: String,
    pub model: String,
    pub shortlist: usize,
}

#[derive(Deserialize)]
pub struct SearchProfile {
    pub candidates: usize,
    pub expanded_candidates: usize,
}

impl Profile {
    pub fn load(path: &Path) -> Result<Self> {
        let profile: Self = toml::from_str(&std::fs::read_to_string(path)?)
            .with_context(|| format!("read search profile {}", path.display()))?;
        loopback(&profile.embedding.url)?;
        loopback(&profile.reranking.url)?;
        ensure!(
            (1..=8192).contains(&profile.embedding.dimension)
                && (1..=128).contains(&profile.embedding.batch_size)
                && !profile.embedding.revision.is_empty()
                && !profile.embedding.model.is_empty()
                && !profile.reranking.model.is_empty()
                && (1..=128).contains(&profile.reranking.shortlist)
                && profile.search.candidates > 0
                && (profile.search.candidates..=1024).contains(&profile.search.expanded_candidates),
            "invalid search limits or model fingerprint"
        );
        Ok(profile)
    }
}

fn loopback(input: &str) -> Result<Url> {
    let url = Url::parse(input).context("invalid model endpoint URL")?;
    let loopback = url
        .host_str()
        .and_then(|host| {
            host.trim_matches(['[', ']'])
                .parse::<std::net::IpAddr>()
                .ok()
        })
        .is_some_and(|ip| ip.is_loopback());
    ensure!(
        loopback && url.scheme() == "http" && url.username().is_empty() && url.password().is_none(),
        "model endpoint must be an explicit loopback HTTP address"
    );
    Ok(url)
}

pub struct ModelClient {
    client: Client,
    rerank_slot: Mutex<()>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<Embedding>,
}

#[derive(Deserialize)]
struct Embedding {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct RerankResponse {
    results: Vec<Rerank>,
}

#[derive(Deserialize)]
struct Rerank {
    index: usize,
    relevance_score: f32,
}

impl ModelClient {
    pub fn new() -> Result<Self> {
        Ok(Self {
            client: Client::builder()
                .no_proxy()
                .redirect(Policy::none())
                .connect_timeout(Duration::from_secs(3))
                .timeout(Duration::from_secs(50))
                .build()?,
            rerank_slot: Mutex::new(()),
        })
    }

    fn post<T: DeserializeOwned>(&self, endpoint: &str, payload: serde_json::Value) -> Result<T> {
        let url = loopback(endpoint)?;
        for attempt in 0..2 {
            let response = match self.client.post(url.clone()).json(&payload).send() {
                Err(error) if attempt == 0 && error.is_connect() => {
                    thread::sleep(Duration::from_millis(300));
                    continue;
                }
                result => result.with_context(|| format!("model request to {endpoint}"))?,
            };
            let status = response.status();
            if attempt == 0 && (status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error())
            {
                thread::sleep(Duration::from_millis(300));
                continue;
            }
            let body: serde_json::Value = response
                .json()
                .with_context(|| format!("non-JSON response from {endpoint} ({status})"))?;
            ensure!(
                status.is_success() && body.get("error").is_none(),
                "model endpoint {endpoint} ({status}): {}",
                body.get("error")
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "invalid response".into())
            );
            return serde_json::from_value(body)
                .with_context(|| format!("wrong model response at {endpoint}"));
        }
        unreachable!("the last attempt returns a result")
    }

    pub fn embed(&self, profile: &EmbeddingProfile, inputs: &[&str]) -> Result<Vec<Vec<f32>>> {
        let response: EmbeddingResponse = self.post(
            &profile.url,
            serde_json::json!({
                "model": profile.model, "input": inputs,
            }),
        )?;
        let mut rows = vec![None; inputs.len()];
        for row in response.data {
            let target = rows
                .get_mut(row.index)
                .ok_or_else(|| anyhow!("embedding index outside batch"))?;
            ensure!(
                target.is_none()
                    && row.embedding.len() == profile.dimension
                    && row.embedding.iter().all(|v| v.is_finite())
                    && row.embedding.iter().any(|v| *v != 0.),
                "invalid or duplicate embedding at index {}",
                row.index
            );
            *target = Some(row.embedding);
        }
        rows.into_iter()
            .map(|row| row.ok_or_else(|| anyhow!("missing embedding")))
            .collect()
    }

    pub fn rerank(
        &self,
        profile: &RerankingProfile,
        query: &str,
        docs: &[&str],
    ) -> Result<Vec<f32>> {
        let _slot = self
            .rerank_slot
            .lock()
            .map_err(|_| anyhow!("reranking worker failed"))?;
        let response: RerankResponse = self.post(
            &profile.url,
            serde_json::json!({
                "model": profile.model, "query": query, "documents": docs, "top_n": docs.len(),
            }),
        )?;
        let mut scores = vec![None; docs.len()];
        for item in response.results {
            let score = scores
                .get_mut(item.index)
                .ok_or_else(|| anyhow!("reranking index outside shortlist"))?;
            ensure!(
                score.is_none() && item.relevance_score.is_finite(),
                "invalid or duplicate reranking score"
            );
            *score = Some(item.relevance_score);
        }
        scores
            .into_iter()
            .map(|n| n.ok_or_else(|| anyhow!("missing reranking score")))
            .collect()
    }

    pub fn prepare(&self, profile: &Profile, use_rerank: bool) -> Result<()> {
        if profile.embedding.check_loaded {
            let url = loopback(&profile.embedding.url)?.join("/api/v1/models")?;
            let catalog: serde_json::Value = self
                .client
                .get(url)
                .send()
                .context("read active LM Studio models")?
                .error_for_status()?
                .json()?;
            let loaded = catalog["models"].as_array().is_some_and(|models| {
                models.iter().any(|model| {
                    model["loaded_instances"]
                        .as_array()
                        .is_some_and(|instances| {
                            instances.iter().any(|instance| {
                                instance["id"].as_str() == Some(&profile.embedding.model)
                            })
                        })
                })
            });
            ensure!(
                loaded,
                "LM Studio model {} has no active instance",
                profile.embedding.model
            );
        }
        let query = embedding_text("Glass tumbler", true);
        self.embed(&profile.embedding, &[&query, "A glass tumbler"])?;
        if use_rerank {
            self.rerank(
                &profile.reranking,
                "Glass tumbler",
                &["A glass tumbler", "A wooden spoon"],
            )
            .context("reranking capability unavailable")?;
        }
        Ok(())
    }
}

pub fn embedding_text(title: &str, query: bool) -> String {
    let title: String = title
        .split('|')
        .next()
        .unwrap_or_default()
        .chars()
        .take(360)
        .collect();
    let title = title.trim();
    if query && !title.is_empty() {
        format!("{QUERY_INSTRUCTION}{title}")
    } else {
        title.to_string()
    }
}

#[derive(Default, Serialize)]
pub struct ModelStats {
    pub hits: usize,
    pub embedded: usize,
    pub batches: usize,
}

pub struct EmbeddingCache {
    store: Connection,
}

impl EmbeddingCache {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let store = Connection::open(path)?;
        store.busy_timeout(Duration::from_secs(15))?;
        store.execute(
            "CREATE TABLE IF NOT EXISTS embeddings(key TEXT PRIMARY KEY, vector BLOB NOT NULL)",
            [],
        )?;
        Ok(Self { store })
    }

    pub fn vectors(
        &mut self,
        model: &ModelClient,
        profile: &EmbeddingProfile,
        texts: &[String],
        stats: &mut ModelStats,
    ) -> Result<Vec<Vec<f32>>> {
        let mut result = vec![Vec::new(); texts.len()];
        let mut missing: BTreeMap<String, (&str, Vec<usize>)> = BTreeMap::new();
        {
            let mut get = self
                .store
                .prepare_cached("SELECT vector FROM embeddings WHERE key=?1")?;
            for (position, text) in texts.iter().enumerate().filter(|(_, s)| !s.is_empty()) {
                let key = format!(
                    "{:x}",
                    Sha256::digest(format!("{}\0{}\0{text}", profile.revision, profile.model))
                );
                use rusqlite::OptionalExtension;
                let saved: Option<Vec<u8>> = get.query_row([&key], |row| row.get(0)).optional()?;
                if let Some(data) = saved.filter(|data| data.len() == profile.dimension * 4) {
                    let vector: Vec<_> = data
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect();
                    if vector.iter().all(|n| n.is_finite()) && vector.iter().any(|n| *n != 0.) {
                        stats.hits += 1;
                        result[position] = vector;
                        continue;
                    }
                }
                missing
                    .entry(key)
                    .or_insert_with(|| (text, Vec::new()))
                    .1
                    .push(position);
            }
        }
        let missing: Vec<_> = missing.into_iter().collect();
        for batch in missing.chunks(profile.batch_size) {
            let texts: Vec<_> = batch.iter().map(|(_, (text, _))| *text).collect();
            let received = model.embed(profile, &texts)?;
            let transaction = self.store.transaction()?;
            for ((key, (_, positions)), vector) in batch.iter().zip(received) {
                let blob: Vec<_> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
                transaction.execute(
                    "INSERT OR REPLACE INTO embeddings VALUES(?1, ?2)",
                    params![key, blob],
                )?;
                for position in positions {
                    result[*position] = vector.clone();
                }
            }
            transaction.commit()?;
            stats.batches += 1;
            stats.embedded += batch.len();
            if stats.batches == 1 || stats.batches.is_multiple_of(32) {
                eprintln!("computed {} new embeddings", stats.embedded);
            }
        }
        Ok(result)
    }
}
