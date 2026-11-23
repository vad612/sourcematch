use crate::{
    analytics,
    input::Offer,
    normalize::{Facts, Lexicon, clean},
    pipeline::matching::{self, Match},
    search::Hit,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, HashSet},
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    locale: String,
    split: String,
    listing: Sample,
    offer: Sample,
    relation: Match,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Sample {
    title: String,
    #[serde(default)]
    attributes: String,
    unit: Option<String>,
    expected: BTreeMap<String, Value>,
}

#[derive(Default, Serialize)]
struct Quality {
    pairs: usize,
    correct_pairs: usize,
    false_acceptances: usize,
    fields: BTreeMap<String, Fields>,
}

#[derive(Default, Serialize)]
struct Fields {
    checked: usize,
    exact: usize,
    missing: usize,
    wrong: usize,
    unexpected: usize,
}

#[derive(Serialize)]
struct Difference {
    case: String,
    locale: String,
    field: String,
    expected: Value,
    actual: Value,
}

#[derive(Default, Serialize)]
struct Report {
    total: Quality,
    by_locale: BTreeMap<String, Quality>,
    differences: Vec<Difference>,
}

pub fn evaluate(
    lexicon: &Lexicon,
    dataset: &Path,
    split: Option<&str>,
    out: Option<&Path>,
) -> Result<()> {
    crate::input::protect_input_files(out, [dataset])?;
    let file = BufReader::new(File::open(dataset)?);
    let mut report = Report::default();
    let mut seen = HashSet::new();
    for (line, record) in file.lines().enumerate() {
        let record = record?;
        if record.trim().is_empty() {
            continue;
        }
        let case: Case = serde_json::from_str(&record)
            .with_context(|| format!("parse matching dataset line {}", line + 1))?;
        ensure!(
            seen.insert(case.id.clone()),
            "duplicate matching case {}",
            case.id
        );
        if split.is_some_and(|split| split != case.split) {
            continue;
        }
        let extract = |sample: &Sample| {
            let unit = sample.unit.as_deref().map(clean);
            lexicon.facts(&sample.title, &sample.attributes, unit.as_deref())
        };
        let listing = extract(&case.listing);
        let offer = Offer {
            id: case.id.clone(),
            title: case.offer.title.clone(),
            facts: extract(&case.offer),
            supplier_id: String::new(),
            moq: None,
            price_tiers: String::new(),
            weight: String::new(),
            package: String::new(),
        };
        let actual = matching::compare(Hit::default(), &listing, &offer, Default::default()).status;
        let local = report.by_locale.entry(case.locale.clone()).or_default();
        for bin in [&mut report.total, &mut *local] {
            bin.pairs += 1;
            bin.correct_pairs += usize::from(actual == case.relation);
            bin.false_acceptances +=
                usize::from(actual == Match::Accepted && actual != case.relation);
        }
        if actual != case.relation {
            report.differences.push(Difference {
                case: case.id.clone(),
                locale: case.locale.clone(),
                field: "relation".into(),
                expected: serde_json::to_value(case.relation)?,
                actual: serde_json::to_value(actual)?,
            });
        }
        for (role, facts, sample) in [
            ("listing", &listing, &case.listing),
            ("offer", &offer.facts, &case.offer),
        ] {
            for (field, expected) in &sample.expected {
                let actual = fact_value(facts, field)?;
                for bin in [&mut report.total, &mut *local] {
                    let metrics = bin.fields.entry(field.clone()).or_default();
                    metrics.checked += 1;
                    if actual == *expected {
                        metrics.exact += 1;
                    } else if actual.is_null() {
                        metrics.missing += 1;
                    } else if expected.is_null() {
                        metrics.unexpected += 1;
                    } else {
                        metrics.wrong += 1;
                    }
                }
                if actual != *expected {
                    report.differences.push(Difference {
                        case: case.id.clone(),
                        locale: case.locale.clone(),
                        field: format!("{role}.{field}"),
                        expected: expected.clone(),
                        actual,
                    });
                }
            }
        }
    }
    ensure!(
        report.total.pairs > 0,
        "matching dataset has no cases for the requested split"
    );
    if let Some(path) = out {
        analytics::write_json(path, &report)?;
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn fact_value(facts: &Facts, field: &str) -> Result<Value> {
    Ok(match field {
        "kind" => serde_json::to_value(&facts.kind)?,
        "category" => serde_json::to_value(&facts.category)?,
        "material" => serde_json::to_value(&facts.material)?,
        "series" => serde_json::to_value(&facts.series)?,
        "units" => serde_json::to_value(facts.units)?,
        "advertised_pack" => serde_json::to_value(facts.advertised_pack)?,
        path if path.starts_with("traits.") => serde_json::to_value(facts.traits.get(&path[7..]))?,
        path if path.starts_with("measures.") => {
            serde_json::to_value(facts.measures.get(&path[9..]))?
        }
        other => anyhow::bail!("unknown matching fact {other}"),
    })
}
