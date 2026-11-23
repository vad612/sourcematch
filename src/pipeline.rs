use crate::{
    economics::{self, Economics, Logistics, Market, Purchase},
    input::{Catalog, Listing, Offer},
    models::{ModelClient, RerankingProfile, embedding_text},
    normalize::Facts,
    search::Hit,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Instant};

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum Match {
    Rejected,
    Uncertain,
    Accepted,
}

#[derive(Debug, Clone, Serialize)]
pub struct Candidate {
    pub offer_id: String,
    pub hit: Hit,
    pub status: Match,
    pub evidence: i32,
    pub shared_measures: usize,
    pub missing: usize,
    pub reason: &'static str,
}

fn compare(candidate: Hit, listing: &Facts, offer: &Offer) -> Candidate {
    let facts = &offer.facts;
    let kind = listing.kind.is_some() && listing.kind == facts.kind;
    let material = listing.material.is_some() && listing.material == facts.material;
    let series = listing.series.is_some() && listing.series == facts.series;
    let mut out = Candidate {
        offer_id: offer.id.clone(),
        hit: candidate,
        status: Match::Uncertain,
        evidence: i32::from(kind) * 5 + i32::from(material) * 3 + i32::from(series) * 3,
        shared_measures: 0,
        missing: 0,
        reason: "insufficient_identity",
    };
    let conflict = [
        (
            listing.kind.is_some() && facts.kind.is_some() && !kind,
            "product",
        ),
        (
            listing.material.is_some() && facts.material.is_some() && !material,
            "material",
        ),
        (
            listing.series.is_some() && facts.series.is_some() && !series,
            "series",
        ),
        (
            listing
                .units
                .zip(facts.units)
                .is_some_and(|(buyer, seller)| buyer % seller != 0),
            "pack",
        ),
    ]
    .into_iter()
    .find_map(|(failed, name)| failed.then_some(name));
    if let Some(reason) = conflict {
        out.status = Match::Rejected;
        out.reason = reason;
        return out;
    }
    let pack_uncertain = listing
        .units
        .zip(facts.advertised_pack)
        .is_some_and(|(buyer, advertised)| buyer % advertised != 0);
    for (axis, wanted) in &listing.measures {
        if let Some(actual) = facts.measures.get(axis) {
            if actual != wanted {
                out.status = Match::Rejected;
                out.reason = "measurement";
                return out;
            }
            out.shared_measures += 1;
            out.evidence += 4;
        } else {
            out.missing += 1;
        }
    }
    for (key, wanted) in &listing.traits {
        if let Some(actual) = facts.traits.get(key) {
            if actual != wanted {
                out.status = Match::Rejected;
                out.reason = "feature";
                return out;
            }
            out.evidence += 2;
        } else {
            out.missing += 1;
        }
    }
    let similarity = out.hit.dense_score.unwrap_or_default();
    let corroborated = kind && material && series && out.missing == 0;
    let semantic = material
        && series
        && out.shared_measures > 0
        && out.missing == 0
        && similarity >= 0.80
        && out.hit.reranked.is_some_and(|score| score >= 0.65);
    if pack_uncertain {
        out.reason = "uncertain_advertised_pack";
    } else if (corroborated || semantic) && listing.units.is_some() && facts.units.is_some() {
        out.status = Match::Accepted;
        out.reason = if corroborated {
            "verified_facts"
        } else {
            "semantic_and_facts"
        };
    }
    out
}

pub struct CatalogIndex<'a> {
    entries: &'a [Catalog],
    linked: HashMap<&'a str, Vec<usize>>,
}

impl<'a> CatalogIndex<'a> {
    pub fn new(entries: &'a [Catalog]) -> Self {
        let mut linked = HashMap::<_, Vec<_>>::new();
        for (id, entry) in entries
            .iter()
            .enumerate()
            .filter(|(_, row)| !row.offer_id.is_empty())
        {
            linked.entry(entry.offer_id.as_str()).or_default().push(id);
        }
        Self { entries, linked }
    }

    fn confirmed(&self, listing: &Facts, offer: &Offer) -> Option<&Catalog> {
        let candidates = self.linked.get(offer.id.as_str())?;
        let mut compatible: Vec<_> = candidates
            .iter()
            .map(|index| &self.entries[*index])
            .filter(|entry| {
                entry.facts.group().is_some()
                    && entry.facts.group() == listing.group()
                    && entry.facts.units == listing.units
                    && offer.facts.units == listing.units
                    && listing
                        .measures
                        .iter()
                        .all(|(name, quantity)| entry.facts.measures.get(name) == Some(quantity))
                    && listing
                        .traits
                        .iter()
                        .all(|(name, value)| entry.facts.traits.get(name) == Some(value))
            })
            .collect();
        compatible.sort_by(|a, b| a.id.cmp(&b.id));
        let first = compatible.first()?;
        compatible
            .iter()
            .all(|other| other.weight == first.weight && other.package == first.package)
            .then_some(*first)
    }
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
enum State {
    Pass,
    Fail,
    Unknown,
}

#[derive(Debug, Serialize)]
struct Gate {
    code: &'static str,
    state: State,
    detail: &'static str,
}

impl Gate {
    fn new(code: &'static str, state: State, detail: &'static str) -> Self {
        Self {
            code,
            state,
            detail,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Output {
    pub listing_id: String,
    pub offer_ids: String,
    pub landed_cost_uzs: String,
    pub unit_margin_uzs: String,
    pub decision: String,
    pub reason: String,
    pub confidence: String,
}

#[derive(Debug, Serialize)]
pub struct Audit {
    pub candidates: Vec<Candidate>,
    pub procurement: Vec<Purchase>,
    gates: Vec<Gate>,
    pub selected: Option<String>,
    pub economics: Option<Economics>,
    pub logistics: Option<Logistics>,
    pub matching_ms: f64,
    pub finance_ms: f64,
    pub reranking_ms: f64,
    pub retrieval_ms: f64,
}

pub struct ResultRow {
    pub output: Output,
    pub audit: Audit,
}

pub struct Context<'a> {
    pub offers: &'a [Offer],
    pub catalog: &'a CatalogIndex<'a>,
    pub market: &'a Market,
    pub reranker: Option<(&'a ModelClient, &'a RerankingProfile)>,
}

impl Context<'_> {
    pub fn process(
        &self,
        id: &str,
        listing: Option<&Listing>,
        hits: Vec<Hit>,
        retrieval_ms: f64,
    ) -> Result<ResultRow> {
        let Some(listing) = listing else {
            return Ok(ResultRow {
                output: Output {
                    listing_id: id.into(), offer_ids: "NONE".into(), landed_cost_uzs: String::new(),
                    unit_margin_uzs: String::new(), decision: "проверить вручную".into(),
                    reason: "NO_MATCH — карточка отсутствует во входной выгрузке, проверить идентификатор".into(),
                    confidence: "0.95".into(),
                },
                audit: Audit {
                    candidates: Vec::new(), procurement: Vec::new(), gates: Vec::new(), selected: None,
                    economics: None, logistics: None, matching_ms: 0., finance_ms: 0., reranking_ms: 0., retrieval_ms,
                },
            });
        };
        let matching_started = Instant::now();
        let mut candidates: Vec<_> = hits
            .into_iter()
            .map(|hit| {
                let offer = &self.offers[hit.offer];
                compare(hit, &listing.facts, offer)
            })
            .collect();
        candidates.sort_by(|a, b| {
            b.evidence
                .cmp(&a.evidence)
                .then(b.hit.rrf().total_cmp(&a.hit.rrf()))
                .then(a.offer_id.cmp(&b.offer_id))
        });
        let mut reranking_ms = 0.;
        if let Some((model, profile)) = self.reranker {
            let shortlist: Vec<_> = candidates
                .iter()
                .enumerate()
                .filter(|(_, candidate)| candidate.status != Match::Rejected)
                .take(profile.shortlist)
                .map(|(i, _)| i)
                .collect();
            if !shortlist.is_empty() {
                let started = Instant::now();
                let query = embedding_text(&listing.title, false);
                let docs: Vec<_> = shortlist
                    .iter()
                    .map(|i| embedding_text(&self.offers[candidates[*i].hit.offer].title, false))
                    .collect();
                let scores = model.rerank(
                    profile,
                    &query,
                    &docs.iter().map(String::as_str).collect::<Vec<_>>(),
                )?;
                reranking_ms = started.elapsed().as_secs_f64() * 1000.;
                for (index, score) in shortlist.into_iter().zip(scores) {
                    let candidate = &mut candidates[index];
                    candidate.hit.reranked = Some(score);
                    if candidate.status == Match::Uncertain {
                        *candidate = compare(
                            candidate.hit.clone(),
                            &listing.facts,
                            &self.offers[candidate.hit.offer],
                        );
                    }
                }
            }
        }
        let mut accepted: Vec<_> = candidates
            .iter()
            .filter(|c| c.status == Match::Accepted)
            .collect();
        accepted.sort_by(|a, b| {
            b.evidence
                .cmp(&a.evidence)
                .then(
                    b.hit
                        .reranked
                        .unwrap_or_default()
                        .total_cmp(&a.hit.reranked.unwrap_or_default()),
                )
                .then(b.hit.rrf().total_cmp(&a.hit.rrf()))
                .then(a.offer_id.cmp(&b.offer_id))
        });
        let plausible = |c: &&Candidate| {
            c.status == Match::Uncertain
                && (c.hit.group || (c.evidence >= 7 && c.shared_measures > 0))
        };
        let unresolved = candidates.iter().filter(plausible).count();
        let matching_ms = (matching_started.elapsed().as_secs_f64() * 1000. - reranking_ms).max(0.);
        let finance_started = Instant::now();

        let procurement: Vec<_> = accepted
            .iter()
            .map(|c| {
                economics::purchase(
                    c.hit.offer,
                    &listing.facts,
                    &self.offers[c.hit.offer],
                    self.market,
                )
            })
            .collect();
        let eligible: Vec<_> = procurement
            .iter()
            .filter(|p| p.eligible != Some(false))
            .collect();
        let winner = eligible
            .iter()
            .find(|p| {
                p.eligible == Some(true)
                    && p.price.is_some_and(|price| {
                        eligible.iter().all(|other| {
                            p.offer_index == other.offer_index
                                || other
                                    .price
                                    .is_some_and(|possible| price.high <= possible.low)
                        })
                    })
            })
            .copied();
        let mut evaluated = Vec::new();
        let tied_upper = eligible
            .iter()
            .filter_map(|p| {
                (p.eligible == Some(true))
                    .then_some(p.price?)
                    .map(|price| price.high)
            })
            .min();
        for variant in eligible.iter().copied().filter(|p| {
            winner.is_none_or(|w| p.offer_index == w.offer_index)
                && tied_upper.is_none_or(|upper| p.price.is_none_or(|price| price.low <= upper))
        }) {
            let offer = &self.offers[variant.offer_index];
            let shipping = economics::logistics(
                &listing.facts,
                offer,
                self.catalog.confirmed(&listing.facts, offer),
            );
            let calculation = economics::economics(listing, variant.price, &shipping, self.market);
            evaluated.push((variant.offer_index, calculation, shipping));
        }
        let economics = winner.and_then(|w| {
            evaluated
                .iter()
                .find(|(index, _, _)| *index == w.offer_index)
        });
        let margin_ceiling_below = !evaluated.is_empty()
            && evaluated.iter().all(|(_, e, _)| {
                e.as_ref()
                    .and_then(|e| e.margin_ceiling.zip(e.threshold_amount))
                    .is_some_and(|(ceiling, threshold)| ceiling < threshold)
            });
        let margin_floor_above = economics
            .and_then(|(_, e, _)| e.as_ref())
            .and_then(|e| e.margin.zip(e.threshold_amount))
            .is_some_and(|(margin, threshold)| margin.low >= threshold);
        let membership = if !accepted.is_empty() {
            Gate::new("NO_MATCH", State::Pass, "подтверждён аналог")
        } else if unresolved == 0 && listing.facts.group().is_some() && !self.offers.is_empty() {
            Gate::new(
                "NO_MATCH",
                State::Fail,
                "не найдено подходящее предложение с указанными характеристиками",
            )
        } else {
            Gate::new(
                "NO_MATCH",
                State::Unknown,
                "недостаточно данных для подтверждения аналога, сверить модель и комплектацию",
            )
        };
        let certified = match listing.facts.category.as_deref() {
            Some(category) if self.market.certified.iter().any(|rule| category == rule) => {
                Gate::new(
                    "CERT",
                    State::Fail,
                    "для фактической категории товара требуется сертификация",
                )
            }
            Some(category) if self.market.commissions.contains_key(category) => {
                Gate::new("CERT", State::Pass, "сертификация по профилю не требуется")
            }
            _ => Gate::new(
                "CERT",
                State::Unknown,
                "уточнить фактическую категорию товара и требования сертификации",
            ),
        };
        let moq = if procurement.iter().any(|p| p.eligible == Some(true)) {
            Gate::new("MOQ", State::Pass, "есть предложение с допустимой партией")
        } else if unresolved == 0
            && !procurement.is_empty()
            && procurement.iter().all(|p| p.eligible == Some(false))
        {
            Gate::new(
                "MOQ",
                State::Fail,
                "минимальная партия превышает лимит у всех подходящих предложений",
            )
        } else {
            Gate::new(
                "MOQ",
                State::Unknown,
                "уточнить единицу поставщика и минимальную партию",
            )
        };
        let margin = if margin_ceiling_below && unresolved == 0 {
            Gate::new(
                "LOW_MARGIN",
                State::Fail,
                "даже верхняя оценка маржи ниже требуемого порога",
            )
        } else if margin_floor_above {
            Gate::new("LOW_MARGIN", State::Pass, "маржа достигает порога")
        } else {
            let detail = if winner.is_none() {
                "не установлена минимальная закупочная цена, уточнить котировки"
            } else {
                "маржа неопределённа, подтвердить цену, вес и упаковку"
            };
            Gate::new("LOW_MARGIN", State::Unknown, detail)
        };
        let demand = demand_gate(listing, self.market);
        let gates = vec![membership, certified, moq, margin, demand];
        let failed = gates.iter().find(|gate| gate.state == State::Fail);
        let unknown = gates.iter().find(|gate| gate.state == State::Unknown);
        let (decision, reason, confidence) = if let Some(gate) = failed {
            (
                "не брать",
                format!("{} — {}", gate.code, gate.detail),
                "0.95",
            )
        } else if let Some(gate) = unknown {
            (
                "проверить вручную",
                format!("{} — {}", gate.code, gate.detail),
                "0.85",
            )
        } else {
            (
                "брать",
                "OK — подтверждены предложение, партия, маржа и спрос".to_owned(),
                if listing.demand.iter().flatten().count() == 1 {
                    "0.75"
                } else {
                    "0.85"
                },
            )
        };
        let selected = winner.map(|purchase| &self.offers[purchase.offer_index].id);
        let ids: Vec<_> = selected
            .into_iter()
            .chain(
                accepted
                    .iter()
                    .filter_map(|c| (selected != Some(&c.offer_id)).then_some(&c.offer_id)),
            )
            .take(3)
            .map(String::as_str)
            .collect();
        let result_economics = economics.and_then(|(_, e, _)| e.clone());
        let render = |range: Option<economics::Range>| {
            range
                .and_then(|r| r.value())
                .map(|v| v.round_dp(2).normalize().to_string())
                .unwrap_or_default()
        };
        Ok(ResultRow {
            output: Output {
                listing_id: listing.id.clone(),
                offer_ids: if ids.is_empty() {
                    "NONE".to_owned()
                } else {
                    ids.join(";")
                },
                landed_cost_uzs: render(result_economics.as_ref().and_then(|e| e.landed)),
                unit_margin_uzs: render(result_economics.as_ref().and_then(|e| e.margin)),
                decision: decision.into(),
                reason,
                confidence: confidence.into(),
            },
            audit: Audit {
                candidates,
                procurement,
                gates,
                selected: selected.cloned(),
                economics: result_economics,
                logistics: economics.map(|(_, _, value)| value.clone()),
                matching_ms,
                finance_ms: finance_started.elapsed().as_secs_f64() * 1000.,
                reranking_ms,
                retrieval_ms,
            },
        })
    }
}

fn demand_gate(listing: &Listing, market: &Market) -> Gate {
    let passes: Vec<_> = listing
        .demand
        .iter()
        .flatten()
        .map(|value| *value >= market.minimum_orders)
        .collect();
    match passes.as_slice() {
        [] => Gate::new(
            "LOW_DEMAND",
            State::Unknown,
            "нет оценок спроса, уточнить ожидаемые заказы",
        ),
        [true, false] | [false, true] => Gate::new(
            "LOW_DEMAND",
            State::Unknown,
            "оценки по разные стороны порога, уточнить ожидаемые заказы",
        ),
        [false] | [false, false] => {
            Gate::new("LOW_DEMAND", State::Fail, "ожидаемые заказы ниже порога")
        }
        _ => Gate::new(
            "LOW_DEMAND",
            State::Pass,
            "ожидаемые заказы достигают порога",
        ),
    }
}
