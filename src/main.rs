mod analytics;
mod economics;
mod input;
mod models;
mod normalize;
mod pipeline;
mod scoring;
mod search;
mod simulation;

use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use models::{EmbeddingCache, ModelClient, ModelStats, Profile, embedding_text};
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Parser)]
#[command(version, about = "Product sourcing and market economics")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Run(Run),
    Prepare {
        #[arg(long, default_value = "profiles/local.toml")]
        search_profile: PathBuf,
        #[arg(long)]
        no_rerank: bool,
    },
    Evaluate {
        #[arg(long)]
        predictions: PathBuf,
        #[arg(long)]
        labels: PathBuf,
        #[arg(long)]
        audit: Option<PathBuf>,
        #[arg(long)]
        out: Option<PathBuf>,
        /// Require every synthetic decision, offer list and monetary field to match its oracle.
        #[arg(long)]
        verify_synthetic: bool,
    },
    Inspect {
        #[arg(long)]
        audit: PathBuf,
        #[arg(long)]
        listing_id: String,
    },
    Simulate {
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        out_dir: PathBuf,
        #[arg(long, default_value = "rules/uzbekistan.toml")]
        rules: PathBuf,
    },
}

#[derive(clap::Args)]
struct Run {
    #[arg(long)]
    listings: PathBuf,
    #[arg(long)]
    offers: PathBuf,
    #[arg(long)]
    catalog: Option<PathBuf>,
    #[arg(long)]
    template: Option<PathBuf>,
    #[arg(long)]
    input_profile: Option<PathBuf>,
    #[arg(long, default_value = "rules/uzbekistan.toml")]
    rules: PathBuf,
    #[arg(long, default_value = "rules/products.tsv")]
    lexicon: PathBuf,
    #[arg(long, default_value = "profiles/local.toml")]
    search_profile: PathBuf,
    /// Enable the versioned portfolio scoring policy and publish its priority queue.
    #[arg(long)]
    scoring_profile: Option<PathBuf>,
    #[arg(long, default_value = "submission.csv")]
    out: PathBuf,
    #[arg(long, default_value = "runs")]
    runs_dir: PathBuf,
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u16).range(1..=64))]
    workers: u16,
    /// Explicit baseline with real embeddings and BM25, without cross-encoder scores.
    #[arg(long)]
    no_rerank: bool,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Run(options) => run(options),
        Command::Prepare {
            search_profile,
            no_rerank,
        } => {
            let profile = Profile::load(&search_profile)?;
            ModelClient::new()?.prepare(&profile, !no_rerank)?;
            println!(
                "ready: embedding={} reranking={}",
                profile.embedding.model,
                if no_rerank {
                    "disabled explicitly"
                } else {
                    profile.reranking.model.as_str()
                }
            );
            Ok(())
        }
        Command::Evaluate {
            predictions,
            labels,
            audit,
            out,
            verify_synthetic,
        } => analytics::evaluate(
            &predictions,
            &labels,
            audit.as_deref(),
            out.as_deref(),
            verify_synthetic,
        ),
        Command::Inspect { audit, listing_id } => analytics::inspect(&audit, &listing_id),
        Command::Simulate {
            seed,
            out_dir,
            rules,
        } => simulation::generate(seed, &out_dir, &rules),
    }
}

#[derive(Serialize)]
struct RunManifest {
    run_id: String,
    status: &'static str,
    cards: usize,
    workers: u16,
    embedding_model: String,
    embedding_revision: String,
    reranking_model: Option<String>,
    reranking_protocol: Option<models::RerankingProtocol>,
    build_version: &'static str,
    inputs: HashMap<String, String>,
    import: input::ImportStats,
    model: ModelStats,
    decisions: std::collections::BTreeMap<String, usize>,
    reasons: std::collections::BTreeMap<String, usize>,
    portfolio: Option<PathBuf>,
    scoring_waves: std::collections::BTreeMap<scoring::Wave, usize>,
    audit: PathBuf,
    output: PathBuf,
    output_sha256: String,
    elapsed_seconds: f64,
    prep_seconds: f64,
    latency_ms: analytics::Latency,
}

fn digest(path: &Path) -> Result<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(std::fs::read(path).with_context(|| format!("read {}", path.display()))?)
    ))
}

fn output_identity(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return Ok(path.canonicalize()?);
    }
    let name = path.file_name().context("output must have a file name")?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    Ok(parent.canonicalize()?.join(name))
}

fn run(options: Run) -> Result<()> {
    let started = Instant::now();
    let run_id = format!(
        "{}-{}",
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos(),
        std::process::id()
    );
    let output = output_identity(&options.out)?;
    let mut inputs = HashMap::new();
    for (name, path) in [
        ("listings", Some(&options.listings)),
        ("offers", Some(&options.offers)),
        ("catalog", options.catalog.as_ref()),
        ("template", options.template.as_ref()),
        ("input_profile", options.input_profile.as_ref()),
        ("rules", Some(&options.rules)),
        ("lexicon", Some(&options.lexicon)),
        ("search_profile", Some(&options.search_profile)),
        ("scoring_profile", options.scoring_profile.as_ref()),
    ] {
        if let Some(path) = path {
            ensure!(
                path.canonicalize()? != output,
                "output would overwrite input {}",
                path.display()
            );
            inputs.insert(name.to_owned(), digest(path)?);
        }
    }
    let market = economics::Market::load(&options.rules)?;
    let scoring = options
        .scoring_profile
        .as_deref()
        .map(scoring::Profile::load)
        .transpose()?;
    let profile = Profile::load(&options.search_profile)?;
    let model = ModelClient::new()?;
    model.prepare(&profile, !options.no_rerank)?;
    let mut stats = input::ImportStats::default();
    let source = input::InputProfile::load(options.input_profile.as_deref())?;
    let lexicon = normalize::Lexicon::read(&std::fs::read_to_string(&options.lexicon)?)?;
    let listings = input::listings(&options.listings, &lexicon, &mut stats, &source.listings)?;
    let offers = input::offers(&options.offers, &lexicon, &mut stats, &source.offers)?;
    ensure!(!offers.is_empty(), "no identifiable offers in input");
    let catalog = options
        .catalog
        .as_ref()
        .map(|path| input::catalog(path, &lexicon, &mut stats, &source.catalog))
        .transpose()?
        .unwrap_or_default();
    let ids: Vec<_> = if let Some(template) = &options.template {
        input::read_with_aliases(template, "listing_id", true, &mut stats, &source.template)?
            .into_iter()
            .map(|row| input::value(&row, "listing_id").to_owned())
            .collect()
    } else {
        listings.iter().map(|listing| listing.id.clone()).collect()
    };
    ensure!(!ids.is_empty(), "there are no listings to process");
    let listing_by_id: HashMap<_, _> = listings.iter().map(|l| (l.id.as_str(), l)).collect();
    std::fs::create_dir_all(&options.runs_dir)?;
    let mut cache = EmbeddingCache::open(&options.runs_dir.join("embeddings.sqlite"))?;
    let mut model_stats = ModelStats::default();
    let supplier_texts: Vec<_> = offers
        .iter()
        .map(|o| embedding_text(&o.title, false))
        .collect();
    let vectors = cache.vectors(
        &model,
        &profile.embedding,
        &supplier_texts,
        &mut model_stats,
    )?;
    let search = search::Index::new(&offers, vectors, profile.embedding.dimension)?;
    let query_texts: Vec<_> = ids
        .iter()
        .map(|id| {
            listing_by_id
                .get(id.as_str())
                .map(|l| embedding_text(&l.title, true))
                .unwrap_or_default()
        })
        .collect();
    let query_vectors =
        cache.vectors(&model, &profile.embedding, &query_texts, &mut model_stats)?;
    let prep_seconds = started.elapsed().as_secs_f64();
    let catalog_index = pipeline::CatalogIndex::new(&catalog);
    let context = pipeline::Context {
        offers: &offers,
        market: &market,
        catalog: &catalog_index,
        reranker: (!options.no_rerank).then_some((&model, &profile.reranking)),
        semantic: profile.search.semantic,
        scoring: scoring.as_ref(),
    };
    let workers = rayon::ThreadPoolBuilder::new()
        .num_threads(usize::from(options.workers))
        .build()?;
    let audit_path = options.runs_dir.join(format!("{run_id}.sqlite"));
    let mut audit = analytics::AuditStore::open(&audit_path)?;
    let mut published = tempfile::NamedTempFile::new_in(output.parent().context("output parent")?)?;
    let mut writer = csv::Writer::from_writer(published.as_file_mut());
    let mut totals = analytics::Totals::default();
    let mut portfolio = analytics::Portfolio::default();
    let mut vectors = query_vectors.into_iter();
    for (index, window) in ids.chunks(64).enumerate() {
        let queries: Vec<_> = vectors.by_ref().take(window.len()).collect();
        let valid: Vec<_> = queries.iter().map(|v| !v.is_empty()).collect();
        let retrieval_started = Instant::now();
        let scores = search.dense_scores(queries)?;
        let dense_share_ms =
            retrieval_started.elapsed().as_secs_f64() * 1000. / window.len() as f64;
        let mut work = Vec::with_capacity(window.len());
        for (offset, id) in window.iter().enumerate() {
            let row_started = Instant::now();
            let listing = listing_by_id.get(id.as_str()).copied();
            let hits = if let Some(listing) = listing {
                let scores = scores.row(offset);
                let scores = scores
                    .as_slice()
                    .context("dense score matrix is not contiguous")?;
                let initial = search.get(
                    &listing.title,
                    &listing.facts,
                    scores,
                    valid[offset],
                    profile.search.candidates,
                )?;
                if listing.facts.group().is_none() || !initial.iter().any(|hit| hit.group) {
                    search.get(
                        &listing.title,
                        &listing.facts,
                        scores,
                        valid[offset],
                        profile.search.expanded_candidates,
                    )?
                } else {
                    initial
                }
            } else {
                Vec::new()
            };
            work.push((
                id,
                listing,
                hits,
                dense_share_ms + row_started.elapsed().as_secs_f64() * 1000.,
            ));
        }
        let rows: Result<Vec<_>> = workers.install(|| {
            work.into_par_iter()
                .map(|(id, listing, hits, millis)| {
                    context
                        .process(id, listing, hits, millis)
                        .with_context(|| format!("process listing {id}"))
                })
                .collect()
        });
        let rows = rows?;
        audit.write(index * 64, &rows)?;
        for row in rows {
            totals.record(&row);
            portfolio.record(&row);
            writer.serialize(&row.output)?;
        }
        let done = (index + 1) * 64;
        if done.is_multiple_of(512) || done >= ids.len() {
            eprintln!("processed {}/{} listings", done.min(ids.len()), ids.len());
        }
    }
    writer.flush()?;
    drop(writer);
    published.as_file().sync_all()?;
    ensure!(totals.count == ids.len(), "wrong output row count");
    let output_sha256 = digest(published.path())?;
    let portfolio_path = scoring
        .as_ref()
        .map(|_| options.runs_dir.join(format!("{run_id}-portfolio.csv")));
    if let Some(path) = &portfolio_path {
        portfolio.write(path)?;
    }
    published.persist(&output)?;
    let manifest = RunManifest {
        run_id: run_id.clone(),
        status: "completed",
        cards: totals.count,
        workers: options.workers,
        embedding_model: profile.embedding.model,
        embedding_revision: profile.embedding.revision,
        reranking_model: (!options.no_rerank).then_some(profile.reranking.model),
        reranking_protocol: (!options.no_rerank).then_some(profile.reranking.protocol),
        build_version: env!("CARGO_PKG_VERSION"),
        inputs,
        import: stats,
        model: model_stats,
        decisions: totals.decisions.clone(),
        reasons: totals.reasons.clone(),
        portfolio: portfolio_path,
        scoring_waves: portfolio.waves,
        latency_ms: totals.finish(),
        audit: audit_path,
        output,
        output_sha256,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        prep_seconds,
    };
    let path = options.runs_dir.join(format!("{run_id}.json"));
    analytics::write_json(&path, &manifest)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

#[cfg(test)]
mod tests;
