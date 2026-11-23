use crate::{
    economics::{self, Economics, Logistics, Market, Purchase, Range},
    input::{Listing, Offer},
};
use anyhow::{Context, Result, ensure};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, path::Path};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub demand_retention: Decimal,
    inventory_charge_rate: Decimal,
    minimum_return: Decimal,
    pub logistics: Option<PeerPolicy>,
    pub demand: Option<DemandPolicy>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PeerPolicy {
    minimum_suppliers: usize,
    maximum_spread: Decimal,
    padding: Decimal,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DemandPolicy {
    pub minimum_sources: usize,
}

impl Profile {
    pub fn load(path: &Path) -> Result<Self> {
        let profile: Self = toml::from_str(&std::fs::read_to_string(path)?)
            .with_context(|| format!("read scoring profile {}", path.display()))?;
        ensure!(
            profile.demand_retention > Decimal::ZERO
                && profile.demand_retention <= Decimal::ONE
                && (Decimal::ZERO..=Decimal::ONE).contains(&profile.inventory_charge_rate)
                && profile.minimum_return >= Decimal::ZERO,
            "invalid portfolio scoring thresholds"
        );
        if let Some(logistics) = &profile.logistics {
            ensure!(
                logistics.minimum_suppliers >= 2
                    && logistics.maximum_spread >= Decimal::ONE
                    && logistics.padding >= Decimal::ONE,
                "invalid logistics peer thresholds"
            );
        }
        if let Some(demand) = &profile.demand {
            ensure!(
                (1..=2).contains(&demand.minimum_sources),
                "invalid demand source count"
            );
        }
        Ok(profile)
    }

    pub fn demand_floor(&self, listing: &Listing) -> Option<Decimal> {
        listing
            .demand
            .iter()
            .flatten()
            .min()?
            .checked_mul(self.demand_retention)
    }

    pub fn project(
        &self,
        listing: &Listing,
        purchase: &Purchase,
        economics: &Economics,
    ) -> Option<Projection> {
        let batch = purchase.batch_in_listing_units?;
        let sold = self.demand_floor(listing)?.min(batch).floor();
        let margin = economics.margin?.low;
        let cost = economics.landed?.high;
        let capital = cost.checked_mul(batch)?;
        let inventory = cost.checked_mul(batch.checked_sub(sold)?)?;
        let contribution = margin
            .checked_mul(sold)?
            .checked_sub(inventory.checked_mul(self.inventory_charge_rate)?)?;
        Some(Projection {
            sold_units: sold,
            batch_units: batch,
            unit_margin_uzs: margin,
            capital_uzs: capital,
            unsold_capital_uzs: inventory,
            contribution_uzs: contribution,
            return_on_capital: contribution.checked_div(capital)?,
        })
    }

    pub fn financially_viable(&self, projection: &Projection) -> bool {
        projection.contribution_uzs > Decimal::ZERO
            && projection.return_on_capital >= self.minimum_return
    }

    pub fn review_potential(
        &self,
        listing: &Listing,
        purchase: &Purchase,
        economics: &Economics,
    ) -> Option<Decimal> {
        purchase.price?;
        let margin = economics.margin_ceiling?;
        if margin < economics.threshold_amount? {
            return None;
        }
        let sales = self
            .demand_floor(listing)?
            .min(purchase.batch_in_listing_units?)
            .floor();
        let potential = sales.checked_mul(margin)?;
        (potential > Decimal::ZERO).then_some(potential)
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Projection {
    pub sold_units: Decimal,
    pub batch_units: Decimal,
    pub unit_margin_uzs: Decimal,
    pub capital_uzs: Decimal,
    pub unsold_capital_uzs: Decimal,
    pub contribution_uzs: Decimal,
    pub return_on_capital: Decimal,
}

#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq, Eq, Ord, PartialOrd)]
#[serde(rename_all = "snake_case")]
pub enum Wave {
    Confirmed,
    ComparableLogistics,
    DemandStress,
    #[default]
    Manual,
}

#[derive(Default, Debug, Serialize)]
pub struct Trace {
    pub wave: Wave,
    pub projection: Option<Projection>,
    pub peer_offer_ids: Vec<String>,
    pub stress_demand: Option<Decimal>,
    pub review_potential_uzs: Option<Decimal>,
}

impl PeerPolicy {
    fn range(
        &self,
        peers: &[(&Offer, Logistics)],
        measure: impl Fn(&Logistics) -> Option<Decimal>,
    ) -> Option<Range> {
        let mut suppliers = HashSet::new();
        let mut values = Vec::new();
        for (offer, logistics) in peers {
            if let Some(value) = measure(logistics) {
                suppliers.insert(&offer.supplier_id);
                values.push(value);
            }
        }
        if suppliers.len() < self.minimum_suppliers {
            return None;
        }
        let low = *values.iter().min()?;
        let high = *values.iter().max()?;
        if high > low.checked_mul(self.maximum_spread)? {
            return None;
        }
        Some(Range {
            low: low.checked_div(self.padding)?,
            high: high.checked_mul(self.padding)?,
        })
    }

    pub fn estimate<'a>(
        &self,
        listing: &Listing,
        offer: &Offer,
        selected: &Logistics,
        candidates: impl IntoIterator<Item = &'a Offer>,
        price: Option<Range>,
        market: &Market,
    ) -> Option<(Economics, Vec<String>)> {
        if offer.facts.group().is_none()
            || (offer.facts.measures.is_empty() && offer.facts.traits.is_empty())
            || offer.supplier_id.is_empty()
            || (selected.kg.is_some() && selected.freight_liters.is_some())
        {
            return None;
        }
        let peers: Vec<_> = candidates
            .into_iter()
            .filter(|other| {
                other.supplier_id != offer.supplier_id
                    && !other.supplier_id.is_empty()
                    && offer.facts.group() == other.facts.group()
                    && offer.facts.units == other.facts.units
                    && offer.facts.measures == other.facts.measures
                    && offer.facts.traits == other.facts.traits
            })
            .map(|other| (other, economics::logistics(&listing.facts, other, None)))
            .filter(|(_, values)| {
                (selected.kg.is_none() && values.kg.is_some())
                    || (selected.freight_liters.is_none() && values.freight_liters.is_some())
            })
            .collect();
        let mass = selected
            .kg
            .map(Range::exact)
            .or_else(|| self.range(&peers, |v| v.kg))?;
        let freight = selected
            .freight_liters
            .map(Range::exact)
            .or_else(|| self.range(&peers, |v| v.freight_liters))?;
        let calculate = |kg, freight_liters| {
            economics::economics(
                listing,
                price,
                &Logistics {
                    kg: Some(kg),
                    freight_liters: Some(freight_liters),
                    ..selected.clone()
                },
                market,
            )
        };
        let low = calculate(mass.low, freight.low)?;
        let high = calculate(mass.high, freight.high)?;
        let low_margin = high.margin?.low;
        let high_margin = low.margin?.high;
        let mut related: Vec<_> = peers.iter().map(|(offer, _)| offer.id.clone()).collect();
        related.sort_unstable();
        Some((
            Economics {
                landed: Some(Range {
                    low: low.landed?.low,
                    high: high.landed?.high,
                }),
                margin: Some(Range {
                    low: low_margin,
                    high: high_margin,
                }),
                margin_ceiling: Some(high_margin),
                threshold_amount: low.threshold_amount,
            },
            related,
        ))
    }
}
