use crate::{
    input::Offer,
    normalize::{Facts, clean},
};
use anyhow::Result;
use ndarray::Array2;
use rusqlite::{Connection, params};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Default, Clone, Debug, Serialize)]
pub struct Hit {
    pub offer: usize,
    pub dense_rank: Option<usize>,
    pub dense_score: Option<f32>,
    pub lexical_rank: Option<usize>,
    pub group: bool,
    pub reranked: Option<f32>,
}

impl Hit {
    pub fn rrf(&self) -> f32 {
        self.dense_rank
            .into_iter()
            .chain(self.lexical_rank)
            .map(|rank| 1.0 / (60.0 + rank as f32))
            .sum::<f32>()
    }
}

pub struct Index {
    dense: Array2<f32>,
    lexical: Connection,
    groups: HashMap<(String, String, String), Vec<usize>>,
    dimension: usize,
}

impl Index {
    pub fn new(offers: &[Offer], vectors: Vec<Vec<f32>>, dimension: usize) -> Result<Self> {
        let dense = matrix(vectors, dimension)?;
        let mut lexical = Connection::open_in_memory()?;
        lexical.execute(
            "CREATE VIRTUAL TABLE offers_fts USING fts5(facts, title)",
            [],
        )?;
        let transaction = lexical.transaction()?;
        let mut groups = HashMap::<_, Vec<_>>::new();
        {
            let mut insert = transaction
                .prepare("INSERT INTO offers_fts(rowid, facts, title) VALUES(?1, ?2, ?3)")?;
            for (offset, offer) in offers.iter().enumerate() {
                insert.execute(params![
                    offset as i64 + 1,
                    offer.facts.keys().join(" "),
                    clean(&offer.title)
                ])?;
                if let Some((kind, material, series)) = offer.facts.group() {
                    groups
                        .entry((kind.to_owned(), material.to_owned(), series.to_owned()))
                        .or_default()
                        .push(offset);
                }
            }
        }
        transaction.commit()?;
        Ok(Self {
            dense,
            lexical,
            groups,
            dimension,
        })
    }

    pub fn dense_scores(&self, vectors: Vec<Vec<f32>>) -> Result<Array2<f32>> {
        Ok(matrix(vectors, self.dimension)?.dot(&self.dense.t()))
    }

    pub fn get(
        &self,
        title: &str,
        facts: &Facts,
        scores: &[f32],
        valid_vector: bool,
        limit: usize,
    ) -> Result<Vec<Hit>> {
        let mut hits = BTreeMap::<usize, Hit>::new();
        if valid_vector {
            let mut ranked: Vec<_> = scores
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, score)| *score != 0.)
                .collect();
            let compare =
                |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
            if ranked.len() > limit {
                ranked.select_nth_unstable_by(limit, compare);
                ranked.truncate(limit);
            }
            ranked.sort_unstable_by(compare);
            for (i, (offer, score)) in ranked.into_iter().enumerate() {
                hits.insert(
                    offer,
                    Hit {
                        offer,
                        dense_rank: Some(i + 1),
                        dense_score: Some(score),
                        ..Hit::default()
                    },
                );
            }
        }
        let title = clean(title);
        let terms: BTreeSet<_> = facts
            .keys()
            .into_iter()
            .chain(
                title
                    .split(|c: char| !c.is_alphanumeric())
                    .filter(|term| term.chars().count() >= 3)
                    .take(16)
                    .map(str::to_owned),
            )
            .collect();
        if !terms.is_empty() {
            let query = terms
                .iter()
                .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
                .collect::<Vec<_>>()
                .join(" OR ");
            let mut statement = self.lexical.prepare_cached("SELECT rowid - 1 FROM offers_fts WHERE offers_fts MATCH ?1 ORDER BY bm25(offers_fts, 3, 1), rowid LIMIT ?2")?;
            for (i, row) in statement
                .query_map(params![query, limit as i64], |row| row.get::<_, i64>(0))?
                .enumerate()
            {
                let offer = usize::try_from(row?)?;
                hits.entry(offer)
                    .or_insert_with(|| Hit {
                        offer,
                        ..Hit::default()
                    })
                    .lexical_rank = Some(i + 1);
            }
        }
        if let Some((kind, material, series)) = facts.group()
            && let Some(group) =
                self.groups
                    .get(&(kind.to_owned(), material.to_owned(), series.to_owned()))
        {
            for &offer in group {
                let hit = hits.entry(offer).or_insert_with(|| Hit {
                    offer,
                    ..Hit::default()
                });
                hit.group = true;
                if valid_vector && hit.dense_score.is_none() {
                    hit.dense_score = scores.get(offer).copied();
                }
            }
        }
        let mut hits: Vec<_> = hits.into_values().collect();
        hits.sort_by(|a, b| b.rrf().total_cmp(&a.rrf()).then(a.offer.cmp(&b.offer)));
        Ok(hits)
    }
}

fn matrix(vectors: Vec<Vec<f32>>, dimension: usize) -> Result<Array2<f32>> {
    let mut matrix: Array2<f32> = Array2::zeros((vectors.len(), dimension));
    for (mut row, vector) in matrix.rows_mut().into_iter().zip(vectors) {
        if vector.is_empty() {
            continue;
        }
        anyhow::ensure!(vector.len() == dimension, "invalid dense index dimension");
        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        anyhow::ensure!(norm.is_finite() && norm > 0., "invalid vector norm");
        for (cell, value) in row.iter_mut().zip(vector) {
            *cell = value / norm;
        }
    }
    Ok(matrix)
}
