use crate::{
    input::{self, ImportStats},
    pipeline::{Output, ResultRow},
};
use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OpenFlags, params};
use rust_decimal::Decimal;
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    io::Write,
    path::Path,
};

pub struct AuditStore(Connection);

impl AuditStore {
    pub fn open(path: &Path) -> Result<Self> {
        let store = Connection::open(path)?;
        store.execute_batch(
            "CREATE TABLE listing_audit (
            ordinal INTEGER PRIMARY KEY, listing_id TEXT NOT NULL UNIQUE,
            decision TEXT NOT NULL, reason TEXT NOT NULL, data TEXT NOT NULL
        );",
        )?;
        Ok(Self(store))
    }

    pub fn write(&mut self, first: usize, rows: &[ResultRow]) -> Result<()> {
        let transaction = self.0.transaction()?;
        {
            let mut insert =
                transaction.prepare("INSERT INTO listing_audit VALUES(?1, ?2, ?3, ?4, ?5)")?;
            for (offset, row) in rows.iter().enumerate() {
                insert.execute(params![
                    i64::try_from(first + offset)?,
                    row.output.listing_id,
                    row.output.decision,
                    row.output.reason,
                    serde_json::to_string(&row.audit)?,
                ])?;
            }
        }
        transaction.commit()?;
        Ok(())
    }
}

#[derive(Default)]
pub struct Totals {
    pub count: usize,
    pub decisions: BTreeMap<String, usize>,
    pub reasons: BTreeMap<String, usize>,
    retrieval: Vec<f64>,
    matching: Vec<f64>,
    finance: Vec<f64>,
    reranking: Vec<f64>,
}

#[derive(Serialize)]
pub struct Latency {
    retrieval_p95: f64,
    matching_p95: f64,
    economics_p95: f64,
    reranking_p95: f64,
}

impl Totals {
    pub fn record(&mut self, row: &ResultRow) {
        self.count += 1;
        *self
            .decisions
            .entry(row.output.decision.clone())
            .or_default() += 1;
        *self
            .reasons
            .entry(
                row.output
                    .reason
                    .split_whitespace()
                    .next()
                    .unwrap_or_default()
                    .to_owned(),
            )
            .or_default() += 1;
        self.retrieval.push(row.audit.retrieval_ms);
        self.matching.push(row.audit.matching_ms);
        self.finance.push(row.audit.finance_ms);
        if row.audit.reranking_ms > 0. {
            self.reranking.push(row.audit.reranking_ms);
        }
    }

    pub fn finish(mut self) -> Latency {
        Latency {
            retrieval_p95: p95(&mut self.retrieval),
            matching_p95: p95(&mut self.matching),
            economics_p95: p95(&mut self.finance),
            reranking_p95: p95(&mut self.reranking),
        }
    }
}

fn p95(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.;
    }
    values.sort_unstable_by(f64::total_cmp);
    values[(values.len() * 95).div_ceil(100) - 1]
}

#[derive(Default, Serialize)]
struct Evaluation {
    cards: usize,
    missing_predictions: usize,
    matches_available: usize,
    top3_hits: usize,
    outside_reference: usize,
    predicted_offers: usize,
    reference_in_retrieval: usize,
    retrieval_checked: usize,
    no_match_references: usize,
    correct_none: usize,
    correct_decisions: usize,
    false_launches: usize,
    missed_launches: usize,
    manual_review: usize,
    margin_values: usize,
    margin_reference_comparisons: usize,
    margin_mae_uzs: Option<Decimal>,
    penalty: Decimal,
    by_expected_reason: BTreeMap<String, Breakdown>,
    synthetic_capital: Option<Capital>,
}

#[derive(Default, Serialize)]
struct Capital {
    approved_uzs: Decimal,
    false_launch_uzs: Decimal,
    unsold_after_one_month_uzs: Decimal,
}

#[derive(Default, Serialize)]
struct Breakdown {
    cards: usize,
    correct_reason: usize,
    manual_review: usize,
    penalty: Decimal,
}

pub fn evaluate(
    predictions: &Path,
    labels: &Path,
    audit: Option<&Path>,
    out: Option<&Path>,
) -> Result<()> {
    let mut outputs = HashMap::new();
    for row in csv::Reader::from_path(predictions)?.deserialize::<Output>() {
        let row = row?;
        ensure!(
            !outputs.contains_key(&row.listing_id),
            "duplicate prediction for {}",
            row.listing_id
        );
        outputs.insert(row.listing_id.clone(), row);
    }
    let mut reference_stats = ImportStats::default();
    let reference = input::read(labels, "listing_id", true, &mut reference_stats)?;
    ensure!(!reference.is_empty(), "evaluation labels are empty");
    let audit = audit
        .map(|path| Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY))
        .transpose()?;
    let mut result = Evaluation::default();
    let mut absolute_error = Decimal::ZERO;
    for row in reference {
        let id = input::value(&row, "listing_id");
        let gold_decision = input::value(&row, "decision");
        let gold_reason = input::value(&row, "reason");
        let capital_inputs = input::value(&row, "landed_cost_uzs")
            .parse::<Decimal>()
            .ok()
            .zip(input::value(&row, "oracle_batch").parse::<Decimal>().ok())
            .zip(input::value(&row, "oracle_demand").parse::<Decimal>().ok());
        if capital_inputs.is_some() {
            result
                .synthetic_capital
                .get_or_insert_with(Capital::default);
        }
        result.cards += 1;
        let expected: HashSet<_> = input::value(&row, "correct_offer_ids")
            .split(';')
            .filter(|&x| !x.is_empty() && x != "NONE")
            .collect();
        if expected.is_empty() {
            result.no_match_references += 1;
        } else {
            result.matches_available += 1;
        }
        let breakdown = result
            .by_expected_reason
            .entry(gold_reason.to_owned())
            .or_default();
        breakdown.cards += 1;
        let Some(output) = outputs.get(id) else {
            result.missing_predictions += 1;
            result.manual_review += 1;
            result.penalty += Decimal::new(3, 1);
            breakdown.manual_review += 1;
            breakdown.penalty += Decimal::new(3, 1);
            continue;
        };
        let actual: Vec<_> = output
            .offer_ids
            .split(';')
            .filter(|&x| !x.is_empty() && x != "NONE")
            .collect();
        result.predicted_offers += actual.len();
        result.outside_reference += actual.iter().filter(|&&id| !expected.contains(id)).count();
        result.top3_hits += usize::from(actual.iter().any(|&id| expected.contains(id)));
        result.correct_none += usize::from(expected.is_empty() && actual.is_empty());
        let decision = output.decision.as_str();
        ensure!(
            matches!(decision, "брать" | "не брать" | "проверить вручную"),
            "invalid prediction decision for {id}"
        );
        if decision == "брать"
            && let Some(((unit_cost, batch), demand)) = capital_inputs
            && let Some(capital) = &mut result.synthetic_capital
        {
            capital.approved_uzs += unit_cost * batch;
            capital.unsold_after_one_month_uzs += unit_cost * (batch - demand).max(Decimal::ZERO);
            if gold_decision == "не брать" {
                capital.false_launch_uzs += unit_cost * batch;
            }
        }
        result.correct_decisions += usize::from(decision == gold_decision);
        breakdown.correct_reason +=
            usize::from(output.reason.split_whitespace().next() == Some(gold_reason));
        let penalty = match (decision, gold_decision) {
            ("брать", "не брать") => {
                result.false_launches += 1;
                Decimal::from(5)
            }
            ("не брать", "брать") => {
                result.missed_launches += 1;
                Decimal::ONE
            }
            ("проверить вручную", _) => {
                result.manual_review += 1;
                breakdown.manual_review += 1;
                Decimal::new(3, 1)
            }
            _ => Decimal::ZERO,
        };
        result.penalty += penalty;
        breakdown.penalty += penalty;
        if let Ok(actual) = output.unit_margin_uzs.parse::<Decimal>() {
            result.margin_values += 1;
            if let Ok(reference) = input::value(&row, "unit_margin_uzs").parse::<Decimal>() {
                absolute_error += (actual - reference).abs();
                result.margin_reference_comparisons += 1;
            }
        }
        if let Some(db) = &audit
            && !expected.is_empty()
        {
            let audit: String = db
                .query_row(
                    "SELECT data FROM listing_audit WHERE listing_id=?1",
                    [id],
                    |row| row.get(0),
                )
                .with_context(|| format!("missing audit row {id}"))?;
            let audit: serde_json::Value = serde_json::from_str(&audit)?;
            result.retrieval_checked += 1;
            result.reference_in_retrieval +=
                usize::from(audit["candidates"].as_array().is_some_and(|candidates| {
                    candidates.iter().any(|candidate| {
                        candidate["offer_id"]
                            .as_str()
                            .is_some_and(|offer| expected.contains(offer))
                    })
                }));
        }
    }
    if result.margin_reference_comparisons != 0 {
        result.margin_mae_uzs =
            Some(absolute_error / Decimal::from(result.margin_reference_comparisons));
    }
    if let Some(path) = out {
        write_json(path, &result)?;
    }
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

pub fn inspect(path: &Path, id: &str) -> Result<()> {
    let database = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let raw: String = database.query_row(
        "SELECT data FROM listing_audit WHERE listing_id=?1",
        [id],
        |row| row.get(0),
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::from_str::<serde_json::Value>(&raw)?)?
    );
    Ok(())
}

pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut temp = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut temp, value)?;
    writeln!(temp)?;
    temp.as_file().sync_all()?;
    temp.persist(path)?;
    Ok(())
}
