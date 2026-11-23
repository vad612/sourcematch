use anyhow::{Context, Result};
use rust_decimal::Decimal;
use sha2::{Digest, Sha256};
use std::path::Path;
use toml::Value;

const SCENARIOS: &[&str] = &[
    "base",
    "fx-shock",
    "missing-demand",
    "new-certification",
    "missing-weight",
    "assembled-pack",
    "range-quote",
    "tariff-boundary",
    "volume-freight",
    "moq-limit",
    "source-disagreement",
    "single-unconfirmed",
    "comparable-logistics",
    "comparable-logistics-outlier",
    "demand-stress-positive",
    "demand-stress-overestimate",
    "portfolio-base",
    "portfolio-fx-up",
    "portfolio-demand-down",
];

#[derive(Clone)]
struct Case {
    size: u16,
    sale_price: u32,
    units: u32,
    minimum: u32,
    quote: [u32; 2],
    weight_grams: Option<u32>,
    hidden_weight_grams: Option<u32>,
    peer_weights: [Option<u32>; 2],
    box_cm: [u32; 3],
    estimates: [Option<u32>; 2],
    actual_orders: u32,
    reviews: Option<u32>,
}

impl Default for Case {
    fn default() -> Self {
        Self {
            size: 20,
            sale_price: 125_000,
            units: 1,
            minimum: 1,
            quote: [2, 2],
            weight_grams: Some(250),
            hidden_weight_grams: None,
            peer_weights: [None, None],
            box_cm: [20, 10, 5],
            estimates: [Some(70), None],
            actual_orders: 70,
            reviews: None,
        }
    }
}

pub fn generate(seed: u64, root: &Path, rules: &Path) -> Result<()> {
    std::fs::create_dir_all(root)?;
    let original: Value = toml::from_str(&std::fs::read_to_string(rules)?)?;
    let base_rate = rate(&original["exchange_rate"])?;
    for name in SCENARIOS {
        let mut rules = original.clone();
        let mut case = Case::default();
        match *name {
            "fx-shock" => {
                rules["exchange_rate"] = Value::String((base_rate * Decimal::from(25)).to_string());
            }
            "missing-demand" => case.estimates = [None, None],
            "new-certification" => {
                let categories = rules["certified"]
                    .as_array_mut()
                    .context("certification list")?;
                if !categories
                    .iter()
                    .any(|value| value.as_str() == Some("Декор"))
                {
                    categories.push(Value::String("Декор".into()));
                }
            }
            "missing-weight" => case.weight_grams = None,
            "assembled-pack" => case.units = 2,
            "range-quote" => case.quote = [2, 3],
            "tariff-boundary" => case.box_cm = [20, 10, 3],
            "volume-freight" => case.box_cm = [40, 25, 20],
            "moq-limit" => {
                let limit: u32 = rules["maximum_moq"]
                    .as_integer()
                    .context("maximum MOQ")?
                    .try_into()?;
                case.minimum = limit.checked_add(1).context("MOQ overflow")?;
            }
            "source-disagreement" => case.estimates = [Some(80), Some(200)],
            "single-unconfirmed" => {
                case.estimates = [None, Some(81)];
                case.reviews = Some(3);
            }
            "comparable-logistics" | "comparable-logistics-outlier" => {
                case.weight_grams = None;
                case.peer_weights = [Some(225), Some(260)];
                case.hidden_weight_grams = Some(if *name == "comparable-logistics" {
                    250
                } else {
                    3_000
                });
            }
            "demand-stress-positive" | "demand-stress-overestimate" => {
                case.estimates = [Some(80), Some(180)];
                case.actual_orders = if *name == "demand-stress-positive" {
                    85
                } else {
                    30
                };
            }
            "portfolio-fx-up" => {
                rules["exchange_rate"] =
                    Value::String((base_rate * Decimal::new(12, 1)).to_string());
            }
            _ => {}
        }
        let cases = if name.starts_with("portfolio-") {
            portfolio(seed, *name == "portfolio-demand-down")
        } else {
            vec![case]
        };
        write_scenario(seed, &root.join(name), &rules, &cases)?;
    }
    crate::analytics::write_json(
        &root.join("manifest.json"),
        &serde_json::json!({
            "seed": seed, "scenarios": SCENARIOS,
            "oracle": "independent input decisions and hidden financial outcomes", "version": 3,
        }),
    )
}

fn portfolio(seed: u64, demand_down: bool) -> Vec<Case> {
    (0u8..32)
        .map(|index| {
            let mut hasher = Sha256::new();
            hasher.update(seed.to_le_bytes());
            hasher.update([index]);
            let noise = hasher.finalize();
            let demand = 35 + u32::from(noise[3] % 95);
            let demand = if demand_down { demand * 4 / 5 } else { demand };
            Case {
                size: 21 + u16::from(index),
                sale_price: 35_000 + 350 * u32::from(noise[0]),
                quote: [1 + u32::from(noise[1] % 15); 2],
                weight_grams: Some(150 + u32::from(noise[2] % 60) * 10),
                estimates: [Some(demand), None],
                actual_orders: demand,
                ..Case::default()
            }
        })
        .collect()
}

fn write_scenario(seed: u64, folder: &Path, rules: &Value, cases: &[Case]) -> Result<()> {
    std::fs::create_dir_all(folder)?;
    let oracle = Oracle::read(rules)?;
    let mut listings = csv::Writer::from_path(folder.join("listings.csv"))?;
    let mut offers = csv::Writer::from_path(folder.join("offers.csv"))?;
    let mut gold = csv::Writer::from_path(folder.join("gold.csv"))?;

    let demand_columns =
        [0, 1].map(|index| cases.iter().any(|case| case.estimates[index].is_some()));
    let mut listing_header = vec!["price_uzs", "title", "listing_id", "reviews"];
    for (index, name) in ["est_orders_a", "est_orders_b"].into_iter().enumerate() {
        if demand_columns[index] {
            listing_header.push(name);
        }
    }
    listings.write_record(listing_header)?;
    offers.write_record([
        "offer_id",
        "title_cn",
        "unit",
        "moq",
        "price_tiers",
        "weight",
        "package_size",
        "supplier_id",
    ])?;
    gold.write_record([
        "listing_id",
        "correct_offer_ids",
        "decision",
        "reason",
        "landed_cost_uzs",
        "unit_margin_uzs",
        "oracle_batch",
        "oracle_demand",
        "oracle_unit_cost_uzs",
        "oracle_unit_margin_uzs",
        "oracle_qualified",
    ])?;
    for (index, case) in cases.iter().enumerate() {
        let id = format!("S{seed}-{index:02}");
        let offer_id = format!("O{seed}-{index:02}");
        let size = case.size;
        let mut title = format!("Ваза, стекло, {size} см, круглый, серия Классик");
        if case.units > 1 {
            title.push_str(&format!(", {} шт", case.units));
        }
        let mut row = vec![
            case.sale_price.to_string(),
            title,
            id.clone(),
            case.reviews.map_or_else(String::new, |n| n.to_string()),
        ];
        for (index, &used) in demand_columns.iter().enumerate() {
            if used {
                row.push(case.estimates[index].map_or_else(String::new, |n| n.to_string()));
            }
        }
        listings.write_record(row)?;
        let [low, high] = case.quote;
        let quote = if low == high {
            format!("¥{low}")
        } else {
            format!("¥{low}-{high}")
        };
        let [a, b, c] = case.box_cm;
        let weight = case
            .weight_grams
            .map_or_else(String::new, |grams| format!("{grams}g"));
        let offer_title = format!("玻璃经典款圆形{size}厘米花瓶");
        let package = format!("{a}x{b}x{c}cm");
        offers.write_record([
            &offer_id,
            &offer_title,
            "件",
            &case.minimum.to_string(),
            &quote,
            &weight,
            &package,
            &format!("supplier-{index}"),
        ])?;
        let mut expected_offers = vec![offer_id.clone()];
        for (position, grams) in case.peer_weights.iter().enumerate() {
            if let Some(grams) = grams {
                let peer = format!("{offer_id}-P{position}");
                offers.write_record([
                    &peer,
                    &offer_title,
                    "件",
                    &case.minimum.to_string(),
                    "¥4",
                    &format!("{grams}g"),
                    &package,
                    &format!("peer-{index}-{position}"),
                ])?;
                expected_offers.push(peer);
            }
        }
        let expected = oracle.expected(case)?;
        gold.write_record([
            id,
            expected_offers.join(";"),
            expected.decision.into(),
            expected.reason.into(),
            expected.landed,
            expected.margin,
            oracle
                .starter
                .max(Decimal::from(case.minimum) / Decimal::from(case.units))
                .to_string(),
            case.actual_orders.to_string(),
            expected.outcome_cost,
            expected.outcome_margin,
            expected.outcome_qualified,
        ])?;
    }
    listings.flush()?;
    offers.flush()?;
    gold.flush()?;
    std::fs::write(folder.join("rules.toml"), toml::to_string_pretty(rules)?)?;
    Ok(())
}

fn rate(value: &Value) -> Result<Decimal> {
    match value {
        Value::Integer(number) => Ok(Decimal::from(*number)),
        Value::Float(number) => Ok(number.to_string().parse()?),
        Value::String(number) => Ok(number.parse()?),
        _ => anyhow::bail!("expected a numeric scenario rule"),
    }
}

struct Oracle {
    fx: Decimal,
    shop_revenue_share: Decimal,
    customs_multiplier: Decimal,
    freight_per_gram: Decimal,
    freight_per_cubic_cm: Decimal,
    margin_threshold: Decimal,
    order_threshold: Decimal,
    demand_guard: Option<(Decimal, Decimal, i64)>,
    certified: bool,
    bands: Vec<(Option<Decimal>, bool, Decimal)>,
    starter: Decimal,
    maximum_moq: Decimal,
}

struct Expected {
    decision: &'static str,
    reason: &'static str,
    landed: String,
    margin: String,
    outcome_cost: String,
    outcome_margin: String,
    outcome_qualified: String,
}

impl Oracle {
    fn read(rules: &Value) -> Result<Self> {
        let mut bands = Vec::new();
        for band in rules["fulfillment"]
            .as_array()
            .context("fulfillment schedule")?
        {
            bands.push((
                band.get("liters").map(rate).transpose()?,
                band.get("upper_inclusive")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                rate(&band["fee"])? + rate(&band["delivery"])?,
            ));
        }
        let demand_guard = if let Some(g) = rules.get("demand_guard") {
            Some((
                rate(&g["max_source_ratio"])?,
                rate(&g["single_source_headroom"])?,
                g["minimum_reviews"]
                    .as_integer()
                    .context("minimum reviews")?,
            ))
        } else {
            None
        };
        Ok(Self {
            fx: rate(&rules["exchange_rate"])?,
            shop_revenue_share: Decimal::ONE
                - rate(&rules["commissions"]["Декор"])?
                - rate(&rules["non_pickup"])?,
            customs_multiplier: (Decimal::ONE + rate(&rules["duties"]["glass"])?)
                * (Decimal::ONE + rate(&rules["import_vat"])?),
            freight_per_gram: rate(&rules["shipping_per_kg"])? / Decimal::from(1000),
            freight_per_cubic_cm: rate(&rules["shipping_per_cubic_meter"])?
                / Decimal::from(1_000_000),
            margin_threshold: rate(&rules["minimum_margin"])?,
            order_threshold: rate(&rules["minimum_orders"])?,
            demand_guard,
            certified: rules["certified"]
                .as_array()
                .context("certification list")?
                .iter()
                .any(|v| v.as_str() == Some("Декор")),
            bands,
            starter: rate(&rules["starter_units"])?,
            maximum_moq: rate(&rules["maximum_moq"])?,
        })
    }

    fn expected(&self, case: &Case) -> Result<Expected> {
        let units = Decimal::from(case.units);
        let volume_cm = case
            .box_cm
            .into_iter()
            .map(Decimal::from)
            .product::<Decimal>();
        let fees: Vec<_> = if case.units == 1 {
            let liters = volume_cm / Decimal::from(1000);
            let band = self
                .bands
                .iter()
                .find(|(cap, inclusive, _)| match cap {
                    Some(cap) => liters < *cap || (*inclusive && liters == *cap),
                    None => true,
                })
                .context("synthetic volume outside configured bands")?;
            vec![band.2]
        } else {
            self.bands.iter().map(|band| band.2).collect()
        };
        let low_fee = *fees.iter().min().context("empty fulfillment schedule")?;
        let high_fee = *fees.iter().max().context("empty fulfillment schedule")?;
        let lower_freight = (volume_cm * self.freight_per_cubic_cm)
            .max(Decimal::from(case.weight_grams.unwrap_or(0)) * self.freight_per_gram);
        let landed = case.quote.map(|quote| {
            (Decimal::from(quote) * self.fx + lower_freight) * units * self.customs_multiplier
        });
        let revenue = Decimal::from(case.sale_price) * self.shop_revenue_share;
        let optimistic = revenue - low_fee - landed[0];
        let pessimistic = case.weight_grams.map(|_| revenue - high_fee - landed[1]);
        let required = Decimal::from(case.sale_price) * self.margin_threshold;
        let sized = Decimal::from(case.minimum) <= self.maximum_moq * units;
        let (demand_min, demand_max) = match case.estimates {
            [Some(a), Some(b)] => (Some(a.min(b)), Some(a.max(b))),
            [Some(a), None] | [None, Some(a)] => (Some(a), Some(a)),
            _ => (None, None),
        };
        let demand_fail = demand_max.is_some_and(|n| Decimal::from(n) < self.order_threshold);
        let corroborated = match (self.demand_guard, case.estimates) {
            (Some((spread, _, _)), [Some(a), Some(b)]) => {
                Decimal::from(a.max(b)) <= Decimal::from(a.min(b)) * spread
            }
            (Some((_, headroom, reviews)), [Some(n), None] | [None, Some(n)]) => {
                !case.reviews.is_some_and(|seen| i64::from(seen) < reviews)
                    || Decimal::from(n) >= self.order_threshold * headroom
            }
            _ => true,
        };
        let demand_known =
            demand_min.is_some_and(|n| Decimal::from(n) >= self.order_threshold) && corroborated;
        let (decision, reason) = if self.certified {
            ("не брать", "CERT")
        } else if !sized {
            ("не брать", "MOQ")
        } else if optimistic < required {
            ("не брать", "LOW_MARGIN")
        } else if demand_fail {
            ("не брать", "LOW_DEMAND")
        } else if pessimistic.is_none_or(|value| value < required) {
            ("проверить вручную", "LOW_MARGIN")
        } else if !demand_known {
            ("проверить вручную", "LOW_DEMAND")
        } else {
            ("брать", "OK")
        };
        let render =
            |value: Option<Decimal>| value.map_or_else(String::new, |v| v.round_dp(2).to_string());
        let outcome = case
            .hidden_weight_grams
            .or(case.weight_grams)
            .filter(|_| sized && case.units == 1 && case.quote[0] == case.quote[1])
            .map(|grams| {
                let freight = (Decimal::from(grams) * self.freight_per_gram)
                    .max(volume_cm * self.freight_per_cubic_cm);
                let cost =
                    (Decimal::from(case.quote[0]) * self.fx + freight) * self.customs_multiplier;
                (cost, revenue - low_fee - cost)
            });
        Ok(Expected {
            decision,
            reason,
            landed: render(
                (sized && case.weight_grams.is_some() && landed[0] == landed[1])
                    .then_some(landed[0]),
            ),
            margin: render(pessimistic.filter(|&value| sized && value == optimistic)),
            outcome_cost: render(outcome.map(|(cost, _)| cost)),
            outcome_margin: render(outcome.map(|(_, margin)| margin)),
            outcome_qualified: outcome.map_or_else(String::new, |(_, margin)| {
                (!self.certified
                    && margin >= required
                    && Decimal::from(case.actual_orders) >= self.order_threshold)
                    .to_string()
            }),
        })
    }
}
