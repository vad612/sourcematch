use crate::{
    input::{Catalog, Offer},
    normalize::Facts,
};
use std::collections::HashMap;

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

    pub(super) fn confirmed(&self, listing: &Facts, offer: &Offer) -> Option<&Catalog> {
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
