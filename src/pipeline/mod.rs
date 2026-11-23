use crate::{
    economics::{self, Economics, Logistics, Market, Purchase},
    input::{Listing, Offer},
    models::{ModelClient, RerankingProfile, SemanticThresholds, embedding_text},
    scoring::{self, Trace, Wave},
    search::Hit,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Instant;

mod catalog;
mod decision;
pub(crate) mod matching;

pub use catalog::CatalogIndex;
use decision::{Gate, State, demand_gate};
use matching::{Candidate, Match, compare};

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
    pub scoring: Option<Trace>,
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
    pub semantic: SemanticThresholds,
    pub scoring: Option<&'a scoring::Profile>,
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
                    scoring: self.scoring.map(|_| Trace::default()),
                },
            });
        };
        let matching_started = Instant::now();
        let mut candidates: Vec<_> = hits
            .into_iter()
            .map(|hit| {
                let offer = &self.offers[hit.offer];
                compare(hit, &listing.facts, offer, self.semantic)
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
                            self.semantic,
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
        let mut scoring = self.scoring.map(|_| Trace {
            wave: if failed.is_some() || unknown.is_none() {
                Wave::Confirmed
            } else {
                Wave::Manual
            },
            ..Trace::default()
        });
        if let (Some(profile), Some(trace), Some(winner), Some((_, exact, shipping))) =
            (self.scoring, &mut scoring, winner, economics)
            && failed.is_none()
        {
            let margin_pass = gates[3].state == State::Pass;
            let demand_pass = gates[4].state == State::Pass;
            trace.stress_demand = profile.demand_floor(listing);
            trace.projection = exact
                .as_ref()
                .and_then(|e| profile.project(listing, winner, e));
            if unknown.is_some() && gates[..3].iter().all(|gate| gate.state == State::Pass) {
                let estimated = (!margin_pass)
                    .then(|| {
                        let policy = profile.logistics.as_ref()?;
                        policy.estimate(
                            listing,
                            &self.offers[winner.offer_index],
                            shipping,
                            accepted
                                .iter()
                                .map(|candidate| &self.offers[candidate.hit.offer]),
                            winner.price,
                            self.market,
                        )
                    })
                    .flatten();
                if let Some((range, peers)) = &estimated {
                    trace.projection = profile.project(listing, winner, range);
                    trace.peer_offer_ids.clone_from(peers);
                }
                let margin_supported = margin_pass
                    || estimated.as_ref().is_some_and(|(range, _)| {
                        range
                            .margin
                            .zip(range.threshold_amount)
                            .is_some_and(|(margin, threshold)| margin.low >= threshold)
                    });
                let demand_supported = demand_pass
                    || profile.demand.as_ref().is_some_and(|policy| {
                        listing.demand.iter().flatten().count() >= policy.minimum_sources
                            && trace
                                .stress_demand
                                .is_some_and(|d| d >= self.market.minimum_orders)
                    });
                if margin_supported
                    && demand_supported
                    && trace
                        .projection
                        .as_ref()
                        .is_some_and(|p| profile.financially_viable(p))
                {
                    trace.wave = if demand_pass {
                        Wave::ComparableLogistics
                    } else {
                        Wave::DemandStress
                    };
                }
            }
        }
        if let (Some(profile), Some(trace)) = (self.scoring, &mut scoring)
            && trace.wave == Wave::Manual
            && trace.projection.is_none()
        {
            trace.review_potential_uzs = evaluated
                .iter()
                .filter_map(|(index, economics, _)| {
                    let purchase = procurement.iter().find(|p| p.offer_index == *index)?;
                    profile.review_potential(listing, purchase, economics.as_ref()?)
                })
                .max();
        }
        let deferred = self.scoring.and_then(|profile| {
            let deferral = profile.deferral.as_ref()?;
            if failed.is_some() {
                return None;
            }
            if deferral.require_confirmed_offer && gates[0].state == State::Unknown {
                return Some(("NO_MATCH", "закупка отложена по профилю риска, аналог пока не подтверждён"));
            }
            if deferral.require_margin_ceiling
                && gates[3].state == State::Unknown
                && margin_ceiling_below
            {
                return Some(("LOW_MARGIN", "закупка отложена по профилю риска, верхняя маржа подтверждённых предложений ниже порога"));
            }
            deferral.demand_reason(listing, self.market).map(|reason| ("LOW_DEMAND", reason))
        });
        if deferred.is_some()
            && let Some(trace) = &mut scoring
        {
            trace.wave = Wave::Deferred;
        }
        let (decision, reason, confidence) = if let Some(gate) = failed {
            (
                "не брать",
                format!("{} — {}", gate.code, gate.detail),
                "0.95",
            )
        } else if let Some((code, reason)) = deferred {
            ("не брать", format!("{code} — {reason}"), "0.70")
        } else if scoring
            .as_ref()
            .is_some_and(|trace| trace.wave == Wave::ComparableLogistics)
        {
            (
                "брать",
                "OK — запас маржи подтверждён в сценарии логистики сопоставимых поставщиков".into(),
                "0.65",
            )
        } else if let Some(trace) = &scoring
            && trace.wave == Wave::DemandStress
        {
            (
                "брать",
                if trace.peer_offer_ids.is_empty() {
                    "OK — спрос проходит порог в консервативном сценарии".into()
                } else {
                    "OK — маржа и спрос проходят пороги в консервативном сценарии".into()
                },
                "0.60",
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
                scoring,
            },
        })
    }
}
