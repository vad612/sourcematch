use crate::normalize::{Facts, Lexicon, clean, decimal, positive};
use anyhow::{Context, Result, ensure};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::Path,
};

type Row = BTreeMap<String, String>;
type Aliases = BTreeMap<String, String>;

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InputProfile {
    pub listings: Aliases,
    pub offers: Aliases,
    pub catalog: Aliases,
    pub template: Aliases,
}

impl InputProfile {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        match path {
            Some(path) => toml::from_str(&std::fs::read_to_string(path)?)
                .with_context(|| format!("read input profile {}", path.display())),
            None => Ok(Self::default()),
        }
    }
}

#[derive(Debug)]
pub struct Listing {
    pub id: String,
    pub title: String,
    pub facts: Facts,
    pub price: Option<Decimal>,
    pub demand: [Option<Decimal>; 2],
    pub reviews: Option<u32>,
}

#[derive(Debug)]
pub struct Offer {
    pub id: String,
    pub supplier_id: String,
    pub title: String,
    pub facts: Facts,
    pub moq: Option<u32>,
    pub price_tiers: String,
    pub weight: String,
    pub package: String,
}

#[derive(Debug)]
pub struct Catalog {
    pub id: String,
    pub offer_id: String,
    pub facts: Facts,
    pub weight: Option<Decimal>,
    pub package: String,
}

#[derive(Default, Serialize, Debug)]
pub struct ImportStats {
    pub missing_keys: usize,
    pub duplicate_ids: usize,
    pub conflicting_fields: usize,
    pub invalid_rows: usize,
}

pub fn read(path: &Path, key: &str, strict: bool, stats: &mut ImportStats) -> Result<Vec<Row>> {
    read_with_aliases(path, key, strict, stats, &Aliases::new())
}

pub fn read_with_aliases(
    path: &Path,
    key: &str,
    strict: bool,
    stats: &mut ImportStats,
    aliases: &Aliases,
) -> Result<Vec<Row>> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("open {}", path.display()))?;
    let headers: Vec<_> = reader
        .headers()?
        .iter()
        .map(|h| {
            let h = h.trim().trim_start_matches('\u{feff}');
            aliases.get(h).map_or_else(|| h.to_owned(), Clone::clone)
        })
        .collect();
    ensure!(
        headers.iter().collect::<HashSet<_>>().len() == headers.len(),
        "duplicate columns in {}",
        path.display()
    );
    ensure!(
        headers.iter().any(|h| h == key),
        "missing key column {key} in {}",
        path.display()
    );
    let mut rows: Vec<Row> = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut conflicts = HashSet::new();
    for record in reader.records() {
        let record = record.with_context(|| format!("invalid CSV in {}", path.display()))?;
        if record.len() > headers.len() {
            stats.invalid_rows += 1;
            ensure!(!strict, "invalid template row in {}", path.display());
        }
        let row: Row = headers
            .iter()
            .enumerate()
            .map(|(i, header)| {
                (
                    header.clone(),
                    record.get(i).unwrap_or_default().trim().to_owned(),
                )
            })
            .collect();
        let id = value(&row, key);
        if id.is_empty() || id.chars().any(char::is_control) {
            stats.missing_keys += 1;
            ensure!(
                !strict,
                "blank or malformed template key in {}",
                path.display()
            );
            continue;
        }
        if let Some(&index) = seen.get(id) {
            stats.duplicate_ids += 1;
            ensure!(!strict, "duplicate template key {id}");
            let original = &mut rows[index];
            for (field, variant) in row {
                let current = original.get_mut(&field).expect("same header set");
                if conflicts.contains(&(index, field.clone())) || variant.is_empty() {
                    continue;
                }
                if current.is_empty() {
                    *current = variant;
                } else if *current != variant {
                    current.clear();
                    conflicts.insert((index, field));
                    stats.conflicting_fields += 1;
                }
            }
        } else {
            seen.insert(id.to_owned(), rows.len());
            rows.push(row);
        }
    }
    Ok(rows)
}

pub fn value<'a>(row: &'a Row, field: &str) -> &'a str {
    row.get(field).map(String::as_str).unwrap_or_default()
}

pub fn listings(
    path: &Path,
    lexicon: &Lexicon,
    stats: &mut ImportStats,
    aliases: &Aliases,
) -> Result<Vec<Listing>> {
    Ok(
        read_with_aliases(path, "listing_id", false, stats, aliases)?
            .into_iter()
            .map(|row| {
                let title = value(&row, "title");
                Listing {
                    id: value(&row, "listing_id").to_owned(),
                    title: title.to_owned(),
                    facts: lexicon.facts(title, value(&row, "attributes"), None),
                    price: positive(value(&row, "price_uzs")),
                    demand: ["est_orders_a", "est_orders_b"]
                        .map(|key| decimal(value(&row, key)).filter(|n| !n.is_sign_negative())),
                    reviews: value(&row, "reviews").parse().ok(),
                }
            })
            .collect(),
    )
}

pub fn offers(
    path: &Path,
    lexicon: &Lexicon,
    stats: &mut ImportStats,
    aliases: &Aliases,
) -> Result<Vec<Offer>> {
    let mut rows: Vec<_> = read_with_aliases(path, "offer_id", false, stats, aliases)?
        .into_iter()
        .map(|row| {
            let title = value(&row, "title_cn");
            let description = clean(value(&row, "description_cn"));
            let material = description
                .split([';', '；'])
                .find(|part| part.trim().starts_with("材质:"))
                .unwrap_or_default();
            let unit = clean(value(&row, "unit"));
            // Descriptions are supplier-controlled. Product identity uses the title and structured terms.
            Offer {
                id: value(&row, "offer_id").to_owned(),
                supplier_id: value(&row, "supplier_id").to_owned(),
                title: title.to_owned(),
                facts: lexicon.facts(title, material, Some(&unit)),
                moq: value(&row, "moq").parse::<u32>().ok().filter(|n| *n > 0),
                price_tiers: value(&row, "price_tiers").to_owned(),
                weight: value(&row, "weight").to_owned(),
                package: value(&row, "package_size").to_owned(),
            }
        })
        .collect();
    rows.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(rows)
}

pub fn catalog(
    path: &Path,
    lexicon: &Lexicon,
    stats: &mut ImportStats,
    aliases: &Aliases,
) -> Result<Vec<Catalog>> {
    Ok(read_with_aliases(path, "ru_sku", false, stats, aliases)?
        .into_iter()
        .map(|row| Catalog {
            id: value(&row, "ru_sku").to_owned(),
            offer_id: value(&row, "offer_id").to_owned(),
            facts: lexicon.facts(value(&row, "title_ru"), "", None),
            weight: positive(value(&row, "unit_weight_kg")),
            package: value(&row, "box_cm").to_owned(),
        })
        .collect())
}
