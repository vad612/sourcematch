use anyhow::{Context, Result};
use rust_decimal::Decimal;
use sha2::{Digest, Sha256};
use std::path::Path;

pub fn generate(seed: u64, root: &Path, rules: &Path) -> Result<()> {
    std::fs::create_dir_all(root)?;
    let original = std::fs::read_to_string(rules)?;
    let mut scenarios = Vec::new();
    let cases = [
        (
            "base",
            "1780",
            false,
            false,
            "17955.84",
            "70794.16",
            "брать",
            "OK",
        ),
        (
            "fx-shock",
            "40000",
            false,
            false,
            "120691.20",
            "-31941.20",
            "не брать",
            "LOW_MARGIN",
        ),
        (
            "missing-demand",
            "1780",
            true,
            false,
            "17955.84",
            "70794.16",
            "проверить вручную",
            "LOW_DEMAND",
        ),
        (
            "new-certification",
            "1780",
            false,
            true,
            "17955.84",
            "70794.16",
            "не брать",
            "CERT",
        ),
    ];
    for (name, rate, no_demand, certify, landed, margin, decision, reason) in cases {
        let id = format!("S{seed}-01");
        let offer_id = format!("O{seed}-01");
        let folder = root.join(name);
        std::fs::create_dir_all(&folder)?;
        let mut listings = csv::Writer::from_path(folder.join("listings.csv"))?;
        if no_demand {
            listings.write_record(["listing_id", "title", "price_uzs", "attributes"])?;
            listings.write_record([
                &id,
                "Ваза, стекло, 20 см, круглый, серия Классик",
                "125000",
                "",
            ])?;
        } else {
            listings.write_record([
                "price_uzs",
                "title",
                "listing_id",
                "est_orders_a",
                "attributes",
            ])?;
            listings.write_record([
                "125000",
                "Ваза, стекло, 20 см, круглый, серия Классик",
                &id,
                "70",
                "",
            ])?;
        }
        listings.flush()?;
        let mut offers = csv::Writer::from_path(folder.join("offers.csv"))?;
        offers.write_record([
            "offer_id",
            "title_cn",
            "unit",
            "moq",
            "price_tiers",
            "weight",
            "package_size",
        ])?;
        offers.write_record([
            &offer_id,
            "玻璃经典款圆形20厘米花瓶",
            "件",
            "1",
            "¥2",
            "0.25kg",
            "20x10x5cm",
        ])?;
        offers.flush()?;
        let mut reference = csv::Writer::from_path(folder.join("gold.csv"))?;
        reference.write_record([
            "listing_id",
            "correct_offer_ids",
            "decision",
            "reason",
            "landed_cost_uzs",
            "unit_margin_uzs",
            "oracle_batch",
            "oracle_demand",
        ])?;
        reference.write_record([
            &id, &offer_id, decision, reason, landed, margin, "300", "70",
        ])?;
        reference.flush()?;
        let mut settings: toml::Value = toml::from_str(&original)?;
        settings["exchange_rate"] = toml::Value::String(rate.into());
        if certify {
            settings["certified"]
                .as_array_mut()
                .expect("certified array")
                .push(toml::Value::String("Декор".into()));
        }
        std::fs::write(
            folder.join("rules.toml"),
            toml::to_string_pretty(&settings)?,
        )?;
        scenarios.push(name);
    }
    for (name, fx_up, demand_down) in [
        ("portfolio-base", false, false),
        ("portfolio-fx-up", true, false),
        ("portfolio-demand-down", false, true),
    ] {
        portfolio(seed, &root.join(name), &original, fx_up, demand_down)?;
        scenarios.push(name);
    }
    crate::analytics::write_json(
        &root.join("manifest.json"),
        &serde_json::json!({
            "seed": seed, "scenarios": scenarios, "oracle": "hand-calculated controls and isolated portfolio oracle",
        }),
    )
}

fn portfolio(
    seed: u64,
    folder: &Path,
    original: &str,
    fx_up: bool,
    demand_down: bool,
) -> Result<()> {
    std::fs::create_dir_all(folder)?;
    let mut rules: toml::Value = toml::from_str(original)?;
    let number = |value: &toml::Value| -> Result<Decimal> {
        if let Some(number) = value.as_integer() {
            return Ok(Decimal::from(number));
        }
        Ok(value
            .as_str()
            .context("expected a scenario rate as decimal text")?
            .parse()?)
    };
    let original_fx = number(&rules["exchange_rate"])?;
    let fx = if fx_up {
        original_fx * Decimal::new(12, 1)
    } else {
        original_fx
    };
    rules["exchange_rate"] = toml::Value::String(fx.to_string());
    let commission = number(&rules["commissions"]["Декор"])?;
    let duty = number(&rules["duties"]["glass"])?;
    let duties_multiplier = (Decimal::ONE + duty) * (Decimal::ONE + number(&rules["import_vat"])?);
    let revenue_share = Decimal::ONE - commission - number(&rules["non_pickup"])?;
    let margin_target = number(&rules["minimum_margin"])?;
    let demand_target = number(&rules["minimum_orders"])?;
    let starter_units = number(&rules["starter_units"])?;
    let per_kg = number(&rules["shipping_per_kg"])?;
    let per_liter = number(&rules["shipping_per_cubic_meter"])? / Decimal::from(1000);
    let fees = rules["fulfillment"]
        .as_array()
        .context("fulfillment schedule")?
        .iter()
        .find(|band| {
            band.get("liters").is_none_or(|liters| {
                number(liters).is_ok_and(|cap| {
                    cap > Decimal::ONE
                        || (cap == Decimal::ONE
                            && band.get("upper_inclusive").and_then(toml::Value::as_bool)
                                == Some(true))
                })
            })
        })
        .context("fulfillment for a 1 L box")?;
    let fulfillment = number(&fees["fee"])? + number(&fees["delivery"])?;
    let certified = rules["certified"]
        .as_array()
        .context("certification list")?
        .iter()
        .any(|category| category.as_str() == Some("Декор"));
    let mut listings = csv::Writer::from_path(folder.join("listings.csv"))?;
    let mut offers = csv::Writer::from_path(folder.join("offers.csv"))?;
    let mut reference = csv::Writer::from_path(folder.join("gold.csv"))?;
    listings.write_record(["listing_id", "title", "price_uzs", "est_orders_a"])?;
    offers.write_record([
        "offer_id",
        "title_cn",
        "unit",
        "moq",
        "price_tiers",
        "weight",
        "package_size",
    ])?;
    reference.write_record([
        "listing_id",
        "correct_offer_ids",
        "decision",
        "reason",
        "landed_cost_uzs",
        "unit_margin_uzs",
        "oracle_batch",
        "oracle_demand",
    ])?;
    for index in 0u8..32 {
        let noise = Sha256::digest([seed.to_le_bytes().as_slice(), &[index]].concat());
        let id = format!("{seed}-{index:02}");
        let offer_id = format!("O{id}");
        let size = 21 + index;
        let shop = Decimal::from(35_000 + 350 * u32::from(noise[0]));
        let quote = Decimal::from(1 + noise[1] % 15);
        let weight = Decimal::new(i64::from(15 + noise[2] % 60), 2);
        let actual_demand = Decimal::from(35 + u32::from(noise[3] % 95));
        let demand = if demand_down {
            (actual_demand * Decimal::new(8, 1)).floor()
        } else {
            actual_demand
        };
        let landed = (quote * fx + (weight * per_kg).max(per_liter)) * duties_multiplier;
        let margin = shop * revenue_share - landed - fulfillment;
        let (decision, reason) = if certified {
            ("не брать", "CERT")
        } else if margin < shop * margin_target {
            ("не брать", "LOW_MARGIN")
        } else if demand < demand_target {
            ("не брать", "LOW_DEMAND")
        } else {
            ("брать", "OK")
        };
        listings.write_record([
            &id,
            &format!("Ваза, стекло, {size} см, круглый, серия Классик"),
            &shop.to_string(),
            &demand.to_string(),
        ])?;
        offers.write_record([
            &offer_id,
            &format!("玻璃经典款圆形{size}厘米花瓶"),
            "件",
            "1",
            &format!("¥{quote}"),
            &format!("{weight}kg"),
            "20x10x5cm",
        ])?;
        reference.write_record([
            &id,
            &offer_id,
            decision,
            reason,
            &landed.round_dp(2).to_string(),
            &margin.round_dp(2).to_string(),
            &starter_units.to_string(),
            &demand.to_string(),
        ])?;
    }
    listings.flush()?;
    offers.flush()?;
    reference.flush()?;
    std::fs::write(folder.join("rules.toml"), toml::to_string_pretty(&rules)?)?;
    Ok(())
}
