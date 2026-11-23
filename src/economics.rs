use crate::input::{Catalog, Listing, Offer};
use crate::normalize::{Facts, clean, dimensions_liters, packed_weight, positive};
use anyhow::{Context, Result, ensure};
use regex::Regex;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path, sync::LazyLock};

#[derive(Deserialize)]
pub struct Market {
    pub exchange_rate: Decimal,
    pub starter_units: u32,
    pub maximum_moq: u32,
    pub shipping_per_kg: Decimal,
    pub shipping_per_cubic_meter: Decimal,
    pub import_vat: Decimal,
    pub non_pickup: Decimal,
    pub minimum_margin: Decimal,
    pub minimum_orders: Decimal,
    pub certified: Vec<String>,
    pub commissions: BTreeMap<String, Decimal>,
    pub duties: BTreeMap<String, Decimal>,
    pub fulfillment: Vec<Fulfillment>,
}

#[derive(Deserialize)]
pub struct Fulfillment {
    pub liters: Option<Decimal>,
    #[serde(default)]
    pub upper_inclusive: bool,
    pub fee: Decimal,
    pub delivery: Decimal,
}

impl Market {
    pub fn load(path: &Path) -> Result<Self> {
        let settings: Self = toml::from_str(&std::fs::read_to_string(path)?)
            .with_context(|| format!("read market rules {}", path.display()))?;
        let nonnegative = |n: Decimal| n >= Decimal::ZERO;
        ensure!(
            settings.exchange_rate > Decimal::ZERO
                && settings.starter_units > 0
                && settings.maximum_moq >= settings.starter_units,
            "invalid exchange rate or procurement quantities"
        );
        ensure!(
            [
                settings.shipping_per_kg,
                settings.shipping_per_cubic_meter,
                settings.minimum_orders
            ]
            .into_iter()
            .all(nonnegative),
            "negative shipping tariff or demand threshold"
        );
        ensure!(
            settings.commissions.values().copied().all(|commission| {
                commission >= Decimal::ZERO
                    && commission
                        .checked_add(settings.non_pickup)
                        .is_some_and(|n| n <= Decimal::ONE)
            }),
            "invalid commission plus non-pickup rate"
        );
        ensure!(
            settings
                .duties
                .values()
                .copied()
                .chain([
                    settings.import_vat,
                    settings.non_pickup,
                    settings.minimum_margin
                ])
                .all(|n| (Decimal::ZERO..=Decimal::ONE).contains(&n)),
            "invalid percentage rate"
        );
        ensure!(
            !settings.fulfillment.is_empty(),
            "empty fulfillment schedule"
        );
        let mut previous = Decimal::ZERO;
        for (index, band) in settings.fulfillment.iter().enumerate() {
            ensure!(
                nonnegative(band.fee) && nonnegative(band.delivery),
                "negative fulfillment charge"
            );
            if let Some(cap) = band.liters {
                ensure!(
                    cap > previous && index + 1 < settings.fulfillment.len(),
                    "invalid fulfillment thresholds"
                );
                previous = cap;
            } else {
                ensure!(
                    index + 1 == settings.fulfillment.len(),
                    "unbounded fulfillment band must be last"
                );
            }
        }
        Ok(settings)
    }

    pub fn fulfillment_charge(&self, liters: Decimal) -> Option<Decimal> {
        let band = self.fulfillment.iter().find(|f| {
            f.liters
                .is_none_or(|cap| liters < cap || (f.upper_inclusive && liters == cap))
        })?;
        band.fee.checked_add(band.delivery)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Range {
    pub low: Decimal,
    pub high: Decimal,
}

impl Range {
    #[cfg(test)]
    pub fn exact(n: Decimal) -> Self {
        Self { low: n, high: n }
    }

    pub fn value(self) -> Option<Decimal> {
        (self.low == self.high).then_some(self.low)
    }

    fn times(self, value: Decimal) -> Option<Self> {
        Some(Self {
            low: self.low.checked_mul(value)?,
            high: self.high.checked_mul(value)?,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Purchase {
    pub offer_index: usize,
    pub eligible: Option<bool>,
    pub supplier_units: Option<u32>,
    pub price: Option<Range>,
}

pub fn purchase(index: usize, listing: &Facts, offer: &Offer, market: &Market) -> Purchase {
    let mut result = Purchase {
        offer_index: index,
        eligible: None,
        supplier_units: None,
        price: None,
    };
    let (Some(buyer), Some(seller), Some(moq)) = (listing.units, offer.facts.units, offer.moq)
    else {
        return result;
    };
    let moq_pieces = u64::from(moq) * u64::from(seller);
    let limit_pieces = u64::from(market.maximum_moq) * u64::from(buyer);
    result.eligible = Some(moq_pieces <= limit_pieces);
    if moq_pieces > limit_pieces {
        return result;
    }
    let wanted = u64::from(market.starter_units) * u64::from(buyer);
    let actual = wanted.max(moq_pieces).div_ceil(u64::from(seller));
    result.supplier_units = u32::try_from(actual).ok();
    if let Some(quantity) = result.supplier_units {
        let conversion = Decimal::from(buyer) / Decimal::from(seller);
        result.price = price_at(&offer.price_tiers, quantity).and_then(|p| p.times(conversion));
    }
    result
}

static TIER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*(≥|>=|>)?\s*(\d+)(?:\s*[-–]\s*(\d+))?\s*[^:：]*[:：]\s*(.+)$").unwrap()
});
static QUOTE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*(?:(?:cny|rmb|[¥￥])\s*)?(\d+(?:[.,]\d+)?)(?:\s*[-–~]\s*[¥￥]?\s*(\d+(?:[.,]\d+)?))?\s*(?:元)?\s*$").unwrap()
});
static FLOOR_TIER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*(\d+)\s*(?:个|件|套|只)\s*(?:起批|以上)\s*(.+)$").unwrap());

fn quote(text: &str) -> Option<Range> {
    let captures = QUOTE.captures(text)?;
    let low = positive(&captures[1])?;
    let high = captures
        .get(2)
        .map_or(Some(low), |n| positive(n.as_str()))?;
    (high >= low).then_some(Range { low, high })
}

pub fn price_at(text: &str, quantity: u32) -> Option<Range> {
    let text = clean(text);
    if let Some(price) = quote(&text) {
        return Some(price);
    }
    if text.contains("起批") || text.contains("以上") {
        let mut floors = BTreeMap::new();
        for part in text.split([';', '；']) {
            let fields = FLOOR_TIER.captures(part)?;
            let start = fields[1].parse::<u32>().ok()?;
            let value = quote(&fields[2])?;
            if floors
                .insert(start, value)
                .is_some_and(|earlier| earlier != value)
            {
                return None;
            }
        }
        return floors
            .range(..=quantity)
            .next_back()
            .map(|(_, value)| *value);
    }
    let mut available = None;
    for part in text.split('|') {
        let fields = TIER.captures(part)?;
        let lower: u32 = fields[2].parse().ok()?;
        let upper = match fields.get(3) {
            Some(value) => Some(value.as_str().parse::<u32>().ok()?),
            None if fields.get(1).is_none() && !fields[0].contains('+') => Some(lower),
            None => None,
        };
        if upper.is_some_and(|end| end < lower) {
            if upper.and_then(|n| n.checked_add(1)) == Some(lower) {
                continue;
            }
            return None;
        }
        let start_matches = quantity > lower
            || (quantity == lower && fields.get(1).is_none_or(|m| m.as_str() != ">"));
        if start_matches && upper.is_none_or(|end| quantity <= end) {
            let price = quote(&fields[4])?;
            if available.is_some_and(|other| other != price) {
                return None;
            }
            available = Some(price);
        }
    }
    available
}

#[derive(Default, Clone, Debug, Serialize)]
pub struct Logistics {
    pub kg: Option<Decimal>,
    pub freight_liters: Option<Decimal>,
    pub retail_liters: Option<Decimal>,
    pub weight_source: Option<String>,
    pub package_source: Option<String>,
}

pub fn logistics(listing: &Facts, offer: &Offer, catalog: Option<&Catalog>) -> Logistics {
    let mut logistics = Logistics::default();
    let (Some(buyer), Some(seller)) = (listing.units, offer.facts.units) else {
        return logistics;
    };
    let conversion = Decimal::from(buyer) / Decimal::from(seller);
    if let Some((mass, count)) = packed_weight(&offer.weight) {
        logistics.kg = if let Some(count) = count {
            let units = if clean(&offer.weight).contains("套/箱") {
                conversion
            } else {
                Decimal::from(buyer)
            };
            mass.checked_div(Decimal::from(count))
                .and_then(|m| m.checked_mul(units))
        } else {
            mass.checked_mul(conversion)
        };
        logistics.weight_source = logistics.kg.map(|_| format!("offer:{}", offer.id));
    }
    if buyer == seller
        && !clean(&offer.package).contains("外箱")
        && let Some(volume) = dimensions_liters(&offer.package)
    {
        logistics.retail_liters = Some(volume);
        logistics.freight_liters = Some(volume);
        logistics.package_source = Some(format!("offer:{}", offer.id));
    }
    if let Some(entry) = catalog {
        if logistics.kg.is_none() {
            logistics.kg = entry.weight;
            logistics.weight_source = entry.weight.map(|_| format!("catalog:{}", entry.id));
        }
        if logistics.retail_liters.is_none()
            && let Some(volume) = dimensions_liters(&entry.package)
        {
            logistics.retail_liters = Some(volume);
            logistics.freight_liters = Some(volume);
            logistics.package_source = Some(format!("catalog:{}", entry.id));
        }
    }
    logistics
}

#[derive(Default, Clone, Debug, Serialize)]
pub struct Economics {
    pub landed: Option<Range>,
    pub margin: Option<Range>,
    pub margin_ceiling: Option<Decimal>,
    pub threshold_amount: Option<Decimal>,
}

pub fn economics(
    listing: &Listing,
    price: Option<Range>,
    logistics: &Logistics,
    market: &Market,
) -> Option<Economics> {
    let duty = listing
        .facts
        .material
        .and_then(|material| market.duties.get(material))
        .copied();
    let weight_charge = logistics
        .kg
        .and_then(|kg| kg.checked_mul(market.shipping_per_kg));
    let volume_charge = logistics.freight_liters.and_then(|liters| {
        liters
            .checked_mul(market.shipping_per_cubic_meter)?
            .checked_div(Decimal::from(1000))
    });
    let shipping_minimum = weight_charge
        .into_iter()
        .chain(volume_charge)
        .max()
        .unwrap_or_default();
    let rate = Decimal::ONE
        .checked_add(duty.unwrap_or_default())?
        .checked_mul(Decimal::ONE.checked_add(market.import_vat)?)?;
    let landed_minimum = price
        .map_or(Decimal::ZERO, |p| p.low)
        .checked_mul(market.exchange_rate)?
        .checked_add(shipping_minimum)?
        .checked_mul(rate)?;
    let landed = match (price, duty, weight_charge, volume_charge) {
        (Some(p), Some(_), Some(weight), Some(volume)) => Some(Range {
            low: landed_minimum,
            high: p
                .high
                .checked_mul(market.exchange_rate)?
                .checked_add(weight.max(volume))?
                .checked_mul(rate)?,
        }),
        _ => None,
    };
    let commission = listing
        .facts
        .category
        .as_deref()
        .and_then(|c| market.commissions.get(c))
        .copied();
    let marketplace_charge = logistics
        .retail_liters
        .and_then(|liters| market.fulfillment_charge(liters));
    let threshold_amount = listing
        .price
        .and_then(|price| price.checked_mul(market.minimum_margin));
    let revenue = listing.price.and_then(|value| {
        value.checked_mul(
            Decimal::ONE
                .checked_sub(commission.unwrap_or_default())?
                .checked_sub(market.non_pickup)?,
        )
    });
    let minimum_fee = market
        .fulfillment
        .iter()
        .filter_map(|band| band.fee.checked_add(band.delivery))
        .min()?;
    let margin_ceiling = revenue.and_then(|income| {
        income
            .checked_sub(landed_minimum)?
            .checked_sub(marketplace_charge.unwrap_or(minimum_fee))
    });
    let margin = match (revenue, commission, landed, marketplace_charge) {
        (Some(income), Some(_), Some(cost), Some(fee)) => Some(Range {
            low: income.checked_sub(cost.high)?.checked_sub(fee)?,
            high: income.checked_sub(cost.low)?.checked_sub(fee)?,
        }),
        _ => None,
    };
    Some(Economics {
        landed,
        margin,
        margin_ceiling,
        threshold_amount,
    })
}
