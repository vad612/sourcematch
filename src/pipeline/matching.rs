use crate::{input::Offer, models::SemanticThresholds, normalize::Facts, search::Hit};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
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

pub(crate) fn compare(
    candidate: Hit,
    listing: &Facts,
    offer: &Offer,
    thresholds: SemanticThresholds,
) -> Candidate {
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
        && similarity >= thresholds.minimum_dense
        && out
            .hit
            .reranked
            .is_some_and(|score| score >= thresholds.minimum_rerank);
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
