use anyhow::{Context, Result, ensure};
use regex::Regex;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, str::FromStr, sync::LazyLock};
use unicode_normalization::UnicodeNormalization;

#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize)]
pub struct Facts {
    pub kind: Option<String>,
    pub category: Option<String>,
    pub material: Option<String>,
    pub series: Option<String>,
    pub measures: BTreeMap<&'static str, Vec<Decimal>>,
    pub traits: BTreeMap<String, String>,
    pub units: Option<u32>,
    pub advertised_pack: Option<u32>,
}

impl Facts {
    pub fn keys(&self) -> Vec<String> {
        let mut keys = Vec::new();
        for (key, value) in &self.measures {
            keys.push(format!(
                "{}{}",
                key,
                value
                    .iter()
                    .map(|v| v.normalize().to_string())
                    .collect::<Vec<_>>()
                    .join("x")
            ));
        }
        for (key, value) in &self.traits {
            keys.push(format!("{key}{value}"));
        }
        for value in [
            self.kind.as_deref(),
            self.material.as_deref(),
            self.series.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            keys.push(value.to_string());
        }
        keys
    }

    pub fn group(&self) -> Option<(&str, &str, &str)> {
        Some((
            self.kind.as_deref()?,
            self.material.as_deref()?,
            self.series.as_deref()?,
        ))
    }
}

pub struct Lexicon {
    kinds: Vec<Kind>,
    terms: Terms,
    pack: PackPatterns,
}

struct Kind {
    id: String,
    category: String,
    aliases: Vec<String>,
    substring_aliases: Vec<String>,
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Terms {
    materials: Vec<Aliases>,
    series: Vec<Aliases>,
    traits: Vec<TraitAliases>,
    products: Vec<Aliases>,
    units: UnitTerms,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Aliases {
    key: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    substring_aliases: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TraitAliases {
    key: String,
    value: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    substring_aliases: Vec<String>,
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct UnitTerms {
    piece: Vec<String>,
    set: Vec<String>,
    offer_pack: Option<String>,
    listing_pack: Option<String>,
    pack_attribute: Option<String>,
}

struct PackPatterns {
    offer: Vec<Regex>,
    listing: Vec<Regex>,
    attributes: Vec<Regex>,
}

fn pack_pattern(name: &str, pattern: Option<&str>) -> Result<Regex> {
    let source = pattern.with_context(|| format!("missing units.{name} expression"))?;
    let pattern = Regex::new(source).with_context(|| format!("invalid units.{name}"))?;
    ensure!(
        pattern.captures_len() == 2,
        "units.{name} needs one numeric capture"
    );
    Ok(pattern)
}

fn pack_patterns(name: &str, base: Option<&str>, overlay: Option<&str>) -> Result<Vec<Regex>> {
    let mut patterns = vec![pack_pattern(name, base)?];
    if let Some(extra) = overlay {
        patterns.push(pack_pattern(name, Some(extra))?);
    }
    Ok(patterns)
}

fn captures_pack<'a>(patterns: &[Regex], text: &'a str) -> Option<regex::Captures<'a>> {
    patterns.iter().find_map(|pattern| pattern.captures(text))
}

fn normalize_aliases(aliases: &mut [String]) -> Result<()> {
    for alias in aliases {
        *alias = clean(alias.trim());
        ensure!(!alias.is_empty(), "empty matching alias");
    }
    Ok(())
}

impl Lexicon {
    #[cfg(test)]
    pub fn read(text: &str) -> Result<Self> {
        Self::read_with_terms(text, include_str!("../rules/matching-terms.toml"), None)
    }

    pub fn read_with_terms(text: &str, terms: &str, extension: Option<&str>) -> Result<Self> {
        let mut kinds = Vec::new();
        for line in text
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        {
            let fields: Vec<_> = line.split('\t').collect();
            anyhow::ensure!(
                fields.len() == 3 && fields.iter().all(|s| !s.trim().is_empty()),
                "invalid product lexicon row: {line}"
            );
            kinds.push(Kind {
                id: fields[0].into(),
                category: fields[1].into(),
                aliases: fields[2]
                    .split('|')
                    .map(clean)
                    .filter(|s| !s.is_empty())
                    .collect(),
                substring_aliases: Vec::new(),
            });
        }
        let mut terms: Terms = toml::from_str(terms).context("read matching terms")?;
        let mut pack_additions = UnitTerms::default();
        if let Some(extension) = extension {
            let mut overlay: Terms = toml::from_str(extension).context("read matching overlay")?;
            terms.materials.append(&mut overlay.materials);
            terms.series.append(&mut overlay.series);
            terms.traits.append(&mut overlay.traits);
            terms.products.append(&mut overlay.products);
            terms.units.piece.append(&mut overlay.units.piece);
            terms.units.set.append(&mut overlay.units.set);
            pack_additions = overlay.units;
        }
        for item in terms.materials.iter_mut().chain(&mut terms.series) {
            ensure!(!item.key.is_empty(), "empty canonical matching key");
            normalize_aliases(&mut item.aliases)?;
            normalize_aliases(&mut item.substring_aliases)?;
        }
        for item in &mut terms.traits {
            ensure!(
                !item.key.is_empty() && !item.value.is_empty(),
                "empty canonical trait"
            );
            normalize_aliases(&mut item.aliases)?;
            normalize_aliases(&mut item.substring_aliases)?;
        }
        for mut item in std::mem::take(&mut terms.products) {
            normalize_aliases(&mut item.aliases)?;
            normalize_aliases(&mut item.substring_aliases)?;
            let kind = kinds
                .iter_mut()
                .find(|kind| kind.id == item.key)
                .with_context(|| format!("unknown product kind {}", item.key))?;
            kind.aliases.append(&mut item.aliases);
            kind.substring_aliases.append(&mut item.substring_aliases);
        }
        normalize_aliases(&mut terms.units.piece)?;
        normalize_aliases(&mut terms.units.set)?;
        ensure!(
            !terms
                .units
                .piece
                .iter()
                .any(|unit| terms.units.set.contains(unit)),
            "an alias cannot represent both a piece and a set"
        );
        let pack = PackPatterns {
            offer: pack_patterns(
                "offer_pack",
                terms.units.offer_pack.as_deref(),
                pack_additions.offer_pack.as_deref(),
            )?,
            listing: pack_patterns(
                "listing_pack",
                terms.units.listing_pack.as_deref(),
                pack_additions.listing_pack.as_deref(),
            )?,
            attributes: pack_patterns(
                "pack_attribute",
                terms.units.pack_attribute.as_deref(),
                pack_additions.pack_attribute.as_deref(),
            )?,
        };
        Ok(Self { kinds, terms, pack })
    }

    pub fn facts(&self, title: &str, attrs: &str, sale_unit: Option<&str>) -> Facts {
        let title = clean(title);
        let attrs = clean(attrs);
        let kind = self
            .kinds
            .iter()
            .flat_map(|kind| {
                matching_aliases(&title, &kind.aliases, &kind.substring_aliases)
                    .map(move |alias| (alias.len(), kind))
            })
            .max_by_key(|(len, _)| *len)
            .map(|(_, kind)| kind);
        let material = find_key(&self.terms.materials, &title)
            .or_else(|| find_key(&self.terms.materials, &attrs));
        let series = find_key(&self.terms.series, &title);
        let mut measures = product_measures(&title);
        for (key, value) in product_measures(&attrs) {
            measures.entry(key).or_insert(value);
        }
        let traits = self
            .terms
            .traits
            .iter()
            .filter(|item| {
                matching_aliases(&title, &item.aliases, &item.substring_aliases)
                    .next()
                    .is_some()
            })
            .map(|item| (item.key.clone(), item.value.clone()))
            .collect();
        let units = match sale_unit {
            Some(unit) if self.terms.units.set.iter().any(|alias| alias == unit) => {
                captures_pack(&self.pack.offer, &title).and_then(|c| whole(&c[1]))
            }
            Some(unit) if self.terms.units.piece.iter().any(|alias| alias == unit) => Some(1),
            Some(_) => None,
            None => captures_pack(&self.pack.attributes, &attrs)
                .or_else(|| captures_pack(&self.pack.listing, &title))
                .and_then(|c| whole(&c[1]))
                .or(Some(1)),
        };
        Facts {
            kind: kind.map(|entry| entry.id.clone()),
            category: kind.map(|entry| entry.category.clone()),
            material,
            series,
            measures,
            traits,
            units,
            advertised_pack: sale_unit
                .and_then(|_| captures_pack(&self.pack.offer, &title))
                .and_then(|capture| whole(&capture[1])),
        }
    }
}

fn find_key(terms: &[Aliases], text: &str) -> Option<String> {
    terms
        .iter()
        .find(|item| {
            matching_aliases(text, &item.aliases, &item.substring_aliases)
                .next()
                .is_some()
        })
        .map(|item| item.key.clone())
}

fn matching_aliases<'a>(
    text: &'a str,
    words: &'a [String],
    substrings: &'a [String],
) -> impl Iterator<Item = &'a String> {
    words
        .iter()
        .filter(move |word| contains_term(text, word))
        .chain(
            substrings
                .iter()
                .filter(move |part| text.contains(part.as_str())),
        )
}

pub fn clean(text: &str) -> String {
    text.nfkc()
        .flat_map(char::to_lowercase)
        .map(|c| match c {
            'ё' => 'е',
            'ʻ' | '’' | '‘' | 'ʼ' | '`' => '\'',
            '×' => 'x',
            c if c.is_control() => ' ',
            c => c,
        })
        .collect::<String>()
}

fn contains_term(text: &str, term: &str) -> bool {
    if term.chars().any(is_cjk) {
        return text.contains(term);
    }
    let same_script_word = |c: char| c.is_alphabetic() && !is_cjk(c);
    text.match_indices(term).any(|(at, part)| {
        !text[..at].chars().next_back().is_some_and(same_script_word)
            && !text[at + part.len()..]
                .chars()
                .next()
                .is_some_and(same_script_word)
    })
}

fn is_cjk(c: char) -> bool {
    ('\u{3400}'..='\u{9fff}').contains(&c)
}

pub fn decimal(input: &str) -> Option<Decimal> {
    let input = input.trim();
    if input.len() > 64 {
        return None;
    }
    Decimal::from_str(&input.replace(',', ".")).ok()
}

pub fn positive(input: &str) -> Option<Decimal> {
    decimal(input).filter(|v| *v > Decimal::ZERO)
}

fn whole(input: &str) -> Option<u32> {
    input.parse().ok().filter(|n| *n > 0)
}

static SIZE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(\d+(?:[.,]\d+)?)\s*[xх*]\s*(\d+(?:[.,]\d+)?)(?:\s*[xх*]\s*(\d+(?:[.,]\d+)?))?\s*(cm|厘米|см|sm|mm|毫米|мм)?").unwrap()
});
static XYZ: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"长(\d+(?:\.\d+)?)宽(\d+(?:\.\d+)?)高(\d+(?:\.\d+)?)").unwrap());
static SCALAR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(\d+(?:[.,]\d+)?)\s*(毫升|厘米|毫米|公斤|千克|ml|мл|cm|см|sm|mm|мм|kg|кг|斤|升|米|m|м|l|л)").unwrap()
});
fn product_measures(text: &str) -> BTreeMap<&'static str, Vec<Decimal>> {
    let mut facts = BTreeMap::new();
    let mut remaining = text.to_owned();
    if let Some(m) = SIZE.captures(text).or_else(|| XYZ.captures(text)) {
        let factor = if m
            .get(4)
            .is_some_and(|s| ["mm", "мм", "毫米"].contains(&s.as_str()))
        {
            Decimal::new(1, 1)
        } else {
            Decimal::ONE
        };
        let mut values: Vec<_> = (1..=3)
            .filter_map(|i| positive(m.get(i)?.as_str()).map(|n| n * factor))
            .collect();
        values.sort_unstable();
        if values.len() >= 2 {
            facts.insert("dimensions", values);
        }
        let full = m.get(0).unwrap();
        remaining.replace_range(full.range(), " ");
    }
    for m in SCALAR.captures_iter(&remaining) {
        let unit = m.get(2).unwrap();
        if remaining[unit.end()..]
            .chars()
            .next()
            .is_some_and(|c| c.is_alphabetic() && !is_cjk(c))
        {
            continue;
        }
        let (kind, factor) = match &m[2] {
            "ml" | "мл" | "毫升" => ("volume", Decimal::ONE),
            "l" | "л" | "升" => ("volume", Decimal::from(1000)),
            "kg" | "кг" | "公斤" | "千克" => ("mass", Decimal::ONE),
            "斤" => ("mass", Decimal::new(5, 1)),
            "m" | "м" | "米" => ("length", Decimal::from(100)),
            "mm" | "мм" | "毫米" => ("thickness", Decimal::ONE),
            _ => ("length", Decimal::ONE),
        };
        if let Some(value) = positive(&m[1]).map(|n| n * factor) {
            facts.entry(kind).or_insert_with(|| vec![value]);
        }
    }
    facts
}

pub fn dimensions_liters(text: &str) -> Option<Decimal> {
    let text = clean(text);
    let capture = SIZE.captures(&text)?;
    let factor = if capture
        .get(4)
        .is_some_and(|s| ["mm", "мм", "毫米"].contains(&s.as_str()))
    {
        Decimal::new(1, 6)
    } else {
        Decimal::new(1, 3)
    };
    positive(capture.get(1)?.as_str())?
        .checked_mul(positive(capture.get(2)?.as_str())?)?
        .checked_mul(positive(capture.get(3)?.as_str())?)?
        .checked_mul(factor)
}

static WEIGHT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(\d+(?:[.,]\d+)?)\s*(kg|公斤|千克|кг|克|斤|g|г)").unwrap());
static BOX_COUNT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(\d+)\s*[个件只套]?\s*/\s*箱").unwrap());

pub fn packed_weight(text: &str) -> Option<(Decimal, Option<u32>)> {
    let text = clean(text);
    let weight = WEIGHT.captures(&text)?;
    let factor = match &weight[2] {
        "克" | "g" | "г" => Decimal::new(1, 3),
        "斤" => Decimal::new(5, 1),
        _ => Decimal::ONE,
    };
    let kilograms = positive(&weight[1])?.checked_mul(factor)?;
    if text.contains("外箱") || text.contains("毛重") {
        return Some((kilograms, Some(whole(&BOX_COUNT.captures(&text)?[1])?)));
    }
    Some((kilograms, None))
}
