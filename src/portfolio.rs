use crate::{
    pipeline::ResultRow,
    scoring::{Projection, Wave},
};
use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OpenFlags};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

#[derive(Default)]
pub struct Portfolio {
    rows: Vec<Priority>,
    pub waves: BTreeMap<Wave, usize>,
}

struct Priority {
    id: String,
    buy: bool,
    reason: String,
    wave: Wave,
    projection: Option<Projection>,
    review_potential: Option<Decimal>,
}

impl Portfolio {
    pub fn record(&mut self, row: &ResultRow) {
        let Some(scoring) = &row.audit.scoring else {
            return;
        };
        *self.waves.entry(scoring.wave).or_default() += 1;
        let decision = row.output.decision.as_str();
        if decision == "не брать" {
            return;
        }
        self.rows.push(Priority {
            id: row.output.listing_id.clone(),
            buy: decision == "брать",
            reason: row.output.reason.clone(),
            wave: scoring.wave,
            projection: scoring.projection,
            review_potential: scoring.review_potential_uzs,
        });
    }

    pub fn write(&mut self, path: &Path) -> Result<()> {
        let financial_key = |row: &Priority| {
            row.projection
                .map(|p| (p.contribution_uzs, p.return_on_capital))
        };
        self.rows.sort_unstable_by(|a, b| {
            b.buy
                .cmp(&a.buy)
                .then_with(|| financial_key(b).cmp(&financial_key(a)))
                .then_with(|| b.review_potential.cmp(&a.review_potential))
                .then_with(|| a.id.cmp(&b.id))
        });
        let mut file = tempfile::NamedTempFile::new_in(path.parent().context("portfolio parent")?)?;
        let mut writer = csv::Writer::from_writer(file.as_file_mut());
        writer.write_record([
            "rank",
            "listing_id",
            "decision",
            "reason",
            "wave",
            "projected_orders",
            "batch_units",
            "unit_margin_floor_uzs",
            "capital_upper_uzs",
            "unsold_capital_upper_uzs",
            "projected_contribution_uzs",
            "projected_return_on_capital",
            "manual_review_potential_uzs",
        ])?;
        let render = |value: Option<Decimal>, places| {
            value.map_or_else(String::new, |n| n.round_dp(places).normalize().to_string())
        };
        for (position, row) in self.rows.iter().enumerate() {
            let projection = row.projection;
            writer.serialize((
                position + 1,
                &row.id,
                if row.buy {
                    "брать"
                } else {
                    "проверить вручную"
                },
                &row.reason,
                row.wave,
                render(projection.map(|p| p.sold_units), 0),
                render(projection.map(|p| p.batch_units), 4),
                render(projection.map(|p| p.unit_margin_uzs), 2),
                render(projection.map(|p| p.capital_uzs), 2),
                render(projection.map(|p| p.unsold_capital_uzs), 2),
                render(projection.map(|p| p.contribution_uzs), 2),
                render(projection.map(|p| p.return_on_capital), 6),
                render(row.review_potential, 2),
            ))?;
        }
        writer.flush()?;
        drop(writer);
        file.persist(path)?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct Plan {
    name: String,
    path: PathBuf,
}

impl Plan {
    pub fn parse(value: &str) -> Result<Self> {
        let (name, path) = value
            .split_once('=')
            .context("use NAME=MANIFEST for each plan")?;
        ensure!(
            !name.is_empty() && !path.is_empty(),
            "empty plan name or manifest path"
        );
        Ok(Self {
            name: name.into(),
            path: path.into(),
        })
    }
}

#[derive(Clone, Copy, Serialize)]
pub struct Stress {
    pub base_sales_retention: Decimal,
    pub sales_retention: Decimal,
    pub cost_multiplier: Decimal,
    pub inventory_charge: Decimal,
}

#[derive(Deserialize)]
struct Manifest {
    inputs: BTreeMap<String, String>,
    #[serde(flatten)]
    inference: Inference,
    audit: PathBuf,
    portfolio: Option<PathBuf>,
    output: PathBuf,
    decisions: BTreeMap<String, usize>,
    cards: usize,
    status: String,
}

#[derive(Deserialize, PartialEq)]
struct Inference {
    embedding_model: String,
    embedding_revision: String,
    reranking_model: Option<String>,
    reranking_protocol: Option<crate::models::RerankingProtocol>,
}

#[derive(Deserialize)]
struct AuditRow {
    scoring: Option<Scoring>,
}

#[derive(Deserialize)]
struct Scoring {
    wave: Wave,
    projection: Option<Projection>,
}

#[derive(Serialize)]
struct Action {
    decision: String,
    reason: String,
    wave: Wave,
    base_contribution_uzs: Option<Decimal>,
    stress_contribution_uzs: Option<Decimal>,
}

#[derive(Default, Serialize)]
struct PlanTotals {
    decisions: BTreeMap<String, usize>,
    buys_with_projection: usize,
    buys_without_projection: usize,
    capital_uzs: Decimal,
    base_contribution_uzs: Decimal,
    stressed_capital_uzs: Decimal,
    stressed_unsold_capital_uzs: Decimal,
    stressed_contribution_uzs: Decimal,
    stressed_gross_losses_uzs: Decimal,
    stressed_loss_positions: usize,
}

#[derive(Serialize)]
struct Comparison {
    baseline: String,
    stress: Stress,
    plans: BTreeMap<String, PlanTotals>,
    changed_cards: BTreeMap<String, BTreeMap<String, Action>>,
}

impl Stress {
    fn outcome(self, projection: Projection) -> Option<(Decimal, Decimal, Decimal)> {
        let sold = projection
            .forecast_orders
            .checked_mul(self.base_sales_retention)?
            .checked_mul(self.sales_retention)?
            .min(projection.batch_units)
            .floor();
        let capital = projection.capital_uzs.checked_mul(self.cost_multiplier)?;
        let landed = capital.checked_div(projection.batch_units)?;
        let landed_increase = projection
            .capital_uzs
            .checked_mul(self.cost_multiplier.checked_sub(Decimal::ONE)?)?
            .checked_div(projection.batch_units)?;
        let inventory = landed.checked_mul(projection.batch_units.checked_sub(sold)?)?;
        let contribution = sold
            .checked_mul(projection.unit_margin_uzs.checked_sub(landed_increase)?)?
            .checked_sub(inventory.checked_mul(self.inventory_charge)?)?;
        Some((capital, inventory, contribution))
    }
}

pub fn compare(plans: &[Plan], stress: Stress, out: Option<&Path>) -> Result<()> {
    ensure!(
        plans.len() >= 2,
        "risk comparison requires two or more plans"
    );
    ensure!(
        (Decimal::ZERO..=Decimal::ONE).contains(&stress.base_sales_retention)
            && (Decimal::ZERO..=Decimal::ONE).contains(&stress.sales_retention)
            && stress.cost_multiplier >= Decimal::ONE
            && (Decimal::ZERO..=Decimal::ONE).contains(&stress.inventory_charge),
        "invalid risk stress scenario"
    );
    let base = Stress {
        sales_retention: Decimal::ONE,
        cost_multiplier: Decimal::ONE,
        ..stress
    };
    let mut comparison = Comparison {
        baseline: plans[0].name.clone(),
        stress,
        plans: BTreeMap::new(),
        changed_cards: BTreeMap::new(),
    };
    let mut shared_run_settings = None;
    let mut shared_ids = None;
    let mut actions: BTreeMap<String, BTreeMap<String, Action>> = BTreeMap::new();
    for plan in plans {
        crate::input::protect_input_files(out, [plan.path.as_path()])?;
        let mut manifest: Manifest = serde_json::from_slice(&std::fs::read(&plan.path)?)
            .with_context(|| format!("read plan manifest {}", plan.path.display()))?;
        ensure!(
            manifest.status == "completed" && manifest.portfolio.is_some(),
            "plan {} is not a completed scored run",
            plan.name
        );
        crate::input::protect_input_files(
            out,
            [
                Some(manifest.audit.as_path()),
                Some(manifest.output.as_path()),
                manifest.portfolio.as_deref(),
            ]
            .into_iter()
            .flatten(),
        )?;
        manifest
            .inputs
            .remove("scoring_profile")
            .with_context(|| format!("plan {} lacks a scoring profile", plan.name))?;
        let settings = (manifest.inputs, manifest.inference);
        if let Some(reference) = &shared_run_settings {
            ensure!(
                *reference == settings,
                "plan {} has different inputs, market or inference mode",
                plan.name
            );
        } else {
            shared_run_settings = Some(settings);
        }
        let db = Connection::open_with_flags(&manifest.audit, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let mut statement =
            db.prepare("SELECT listing_id, decision, reason, data FROM listing_audit")?;
        let mut rows = statement.query([])?;
        let mut ids = BTreeSet::new();
        let mut totals = PlanTotals::default();
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let decision: String = row.get(1)?;
            let reason: String = row.get(2)?;
            let data: String = row.get(3)?;
            let audit: AuditRow = serde_json::from_str(&data)?;
            let scoring = audit
                .scoring
                .with_context(|| format!("missing scoring trace for {id}"))?;
            *totals.decisions.entry(decision.clone()).or_default() += 1;
            let mut action = Action {
                decision,
                reason,
                wave: scoring.wave,
                base_contribution_uzs: None,
                stress_contribution_uzs: None,
            };
            if action.decision == "брать" {
                if let Some(projection) = scoring.projection {
                    let (capital, _, contribution) = base
                        .outcome(projection)
                        .context("base portfolio arithmetic overflow")?;
                    let (stress_capital, inventory, downside) = stress
                        .outcome(projection)
                        .context("stress portfolio arithmetic overflow")?;
                    totals.buys_with_projection += 1;
                    totals.capital_uzs += capital;
                    totals.base_contribution_uzs += contribution;
                    totals.stressed_capital_uzs += stress_capital;
                    totals.stressed_unsold_capital_uzs += inventory;
                    totals.stressed_contribution_uzs += downside;
                    totals.stressed_gross_losses_uzs += (-downside).max(Decimal::ZERO);
                    totals.stressed_loss_positions += usize::from(downside < Decimal::ZERO);
                    action.base_contribution_uzs = Some(contribution.round_dp(2));
                    action.stress_contribution_uzs = Some(downside.round_dp(2));
                } else {
                    totals.buys_without_projection += 1;
                }
            }
            ids.insert(id.clone());
            actions
                .entry(id)
                .or_default()
                .insert(plan.name.clone(), action);
        }
        ensure!(
            ids.len() == manifest.cards && totals.decisions == manifest.decisions,
            "plan {} audit row or decision counts differ from its manifest",
            plan.name
        );
        if let Some(common) = &shared_ids {
            ensure!(
                *common == ids,
                "plan {} has different listing IDs",
                plan.name
            );
        } else {
            shared_ids = Some(ids);
        }
        ensure!(
            comparison.plans.insert(plan.name.clone(), totals).is_none(),
            "duplicate plan {}",
            plan.name
        );
    }
    comparison.changed_cards = actions
        .into_iter()
        .filter(|(_, actions)| {
            actions
                .values()
                .any(|action| action.decision != actions[&comparison.baseline].decision)
        })
        .collect();
    if let Some(path) = out {
        crate::analytics::write_json(path, &comparison)?;
    }
    println!("{}", serde_json::to_string_pretty(&comparison)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stress_revalues_purchase_capital_sales_and_inventory_together() {
        let amount = |value| Decimal::from(value);
        let projection = Projection {
            forecast_orders: amount(100),
            sold_units: amount(25),
            batch_units: amount(100),
            unit_margin_uzs: amount(500),
            capital_uzs: amount(100_000),
            unsold_capital_uzs: amount(75_000),
            contribution_uzs: amount(11_000),
            return_on_capital: Decimal::new(11, 2),
        };
        let stress = Stress {
            base_sales_retention: Decimal::new(8, 1),
            sales_retention: Decimal::new(75, 2),
            cost_multiplier: Decimal::new(115, 2),
            inventory_charge: Decimal::new(2, 2),
        };
        assert_eq!(
            stress.outcome(projection),
            Some((amount(115_000), amount(46_000), amount(20_080)))
        );
    }
}
