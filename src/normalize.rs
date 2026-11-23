use regex::Regex;
use rust_decimal::Decimal;
use serde::Serialize;
use std::{collections::BTreeMap, str::FromStr, sync::LazyLock};
use unicode_normalization::UnicodeNormalization;

#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize)]
pub struct Facts {
    pub kind: Option<String>,
    pub category: Option<String>,
    pub material: Option<&'static str>,
    pub series: Option<&'static str>,
    pub measures: BTreeMap<&'static str, Vec<Decimal>>,
    pub traits: BTreeMap<&'static str, &'static str>,
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
        for value in [self.kind.as_deref(), self.material, self.series]
            .into_iter()
            .flatten()
        {
            keys.push(value.to_string());
        }
        keys
    }

    pub fn group(&self) -> Option<(&str, &str, &str)> {
        Some((self.kind.as_deref()?, self.material?, self.series?))
    }
}

pub struct Lexicon(Vec<Kind>);

struct Kind {
    id: String,
    category: String,
    aliases: Vec<String>,
}

impl Lexicon {
    pub fn read(text: &str) -> anyhow::Result<Self> {
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
            });
        }
        Ok(Self(kinds))
    }

    pub fn facts(&self, title: &str, attrs: &str, sale_unit: Option<&str>) -> Facts {
        let title = clean(title);
        let attrs = clean(attrs);
        let kind = self
            .0
            .iter()
            .flat_map(|kind| {
                kind.aliases
                    .iter()
                    .filter(|alias| contains_term(&title, alias))
                    .map(move |alias| (alias.len(), kind))
            })
            .max_by_key(|(len, _)| *len)
            .map(|(_, kind)| kind);
        let material = find_material(&title).or_else(|| find_material(&attrs));
        let series = TERMS_SERIES
            .iter()
            .find(|(_, aliases)| aliases.iter().any(|alias| contains_term(&title, alias)))
            .map(|(name, _)| *name);
        let mut measures = product_measures(&title);
        for (key, value) in product_measures(&attrs) {
            measures.entry(key).or_insert(value);
        }
        let traits = TRAITS
            .iter()
            .filter_map(|(group, value, aliases)| {
                aliases
                    .iter()
                    .any(|alias| contains_term(&title, alias))
                    .then_some((*group, *value))
            })
            .collect();
        let units = match sale_unit {
            Some("套" | "set" | "набор") => {
                SELLER_SET.captures(&title).and_then(|c| whole(&c[1]))
            }
            Some("件" | "个" | "只" | "piece" | "pcs" | "шт") => Some(1),
            Some(_) => None,
            None => SIZED_LISTING
                .captures(&attrs)
                .or_else(|| LISTING_SET.captures(&title))
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
                .and_then(|_| SELLER_SET.captures(&title))
                .and_then(|capture| whole(&capture[1])),
        }
    }
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

type Aliases = (&'static str, &'static [&'static str]);

const MATERIALS: &[Aliases] = &[
    (
        "steel",
        &[
            "нержавеющая сталь",
            "zanglamaydigan po'lat",
            "不锈钢",
            "stainless steel",
        ],
    ),
    ("abs", &["abs"]),
    ("eva", &["eva"]),
    ("plastic", &["пластик", "plastik", "塑料", "plastic"]),
    ("silicone", &["силикон", "silikon", "硅胶", "silicone"]),
    ("bamboo", &["бамбук", "bambuk", "竹制", "竹子", "bamboo"]),
    ("wood", &["дерево", "yog'och", "实木", "木质", "wood"]),
    ("metal", &["металл", "metall", "铁艺", "金属", "metal"]),
    (
        "polyester",
        &["полиэстер", "полиэфир", "poliester", "涤纶", "polyester"],
    ),
    ("cotton", &["хлопок", "paxta", "纯棉", "棉布", "cotton"]),
    ("glass", &["стекло", "shisha", "玻璃", "glass"]),
    ("ceramic", &["керамика", "keramika", "陶瓷", "ceramic"]),
];

fn find_material(text: &str) -> Option<&'static str> {
    MATERIALS
        .iter()
        .find(|(_, terms)| terms.iter().any(|term| contains_term(text, term)))
        .map(|(key, _)| *key)
}

const TERMS_SERIES: &[Aliases] = &[
    ("classic", &["классик", "klassik", "classic", "经典款"]),
    ("lite", &["лайт", "lite", "简约款"]),
    ("max", &["макс", "max", "豪华款"]),
];

const TRAITS: &[(&str, &str, &[&str])] = &[
    ("lid", "yes", &["с крышкой", "qopqoqli", "带盖"]),
    ("lid", "no", &["без крышки", "qopqoqsiz", "无盖"]),
    ("handles", "yes", &["с ручками", "dastali", "带把手"]),
    ("handles", "no", &["без ручек", "dastasiz", "无把手"]),
    (
        "mount",
        "wall",
        &["настенный", "devorga osiladigan", "壁挂式"],
    ),
    (
        "mount",
        "floor",
        &["напольный", "polga qo'yiladigan", "落地式"],
    ),
    ("levels", "one", &["одноярусный", "bir qavatli", "单层"]),
    ("levels", "two", &["двухъярусный", "ikki qavatli", "双层"]),
    ("seal", "yes", &["герметичный", "germetik", "密封"]),
    ("seal", "no", &["обычный", "oddiy", "普通款"]),
    ("shape", "round", &["круглый", "dumaloq", "圆形"]),
    ("shape", "square", &["квадратный", "kvadrat", "方形"]),
    (
        "shape",
        "rect",
        &["прямоугольный", "to'rtburchak", "长方形"],
    ),
    (
        "fold",
        "yes",
        &["складной", "buklanadigan", "yig'iladigan", "可折叠"],
    ),
    ("fold", "no", &["каркасный", "karkasli", "固定款"]),
    (
        "surface",
        "non-slip",
        &["противоскользящий", "sirpanmaydigan", "防滑"],
    ),
    ("pattern", "print", &["с принтом", "naqshli", "印花"]),
    ("pattern", "plain", &["однотонный", "bir rangli", "纯色"]),
    (
        "operation",
        "electronic",
        &["электронный", "elektron", "电子"],
    ),
    (
        "operation",
        "mechanical",
        &["механический", "mexanik", "机械"],
    ),
    ("switch", "touch", &["сенсорный", "sensorli", "触摸款"]),
    ("switch", "button", &["кнопочный", "tugmali", "按键款"]),
    ("power", "usb", &["usb", "usb充电"]),
];

static SIZE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(\d+(?:[.,]\d+)?)\s*[xх*]\s*(\d+(?:[.,]\d+)?)(?:\s*[xх*]\s*(\d+(?:[.,]\d+)?))?\s*(cm|厘米|см|sm|mm|毫米|мм)?").unwrap()
});
static XYZ: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"长(\d+(?:\.\d+)?)宽(\d+(?:\.\d+)?)高(\d+(?:\.\d+)?)").unwrap());
static SCALAR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(\d+(?:[.,]\d+)?)\s*(毫升|厘米|毫米|公斤|千克|ml|мл|cm|см|sm|mm|мм|kg|кг|斤|升|米|m|м|l|л)").unwrap()
});
static SELLER_SET: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(\d+)\s*(?:个\s*(?:/\s*套|装)|件\s*(?:/\s*套|装)|只装|pcs\s*/\s*set)").unwrap()
});
static SIZED_LISTING: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:количество в наборе|to'plamdagi soni|pack size)\s*:\s*(\d+)").unwrap()
});
static LISTING_SET: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(\d+)\s*(?:шт|dona|pcs)\b").unwrap());

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
