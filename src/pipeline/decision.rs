use crate::{economics::Market, input::Listing};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub(super) enum State {
    Pass,
    Fail,
    Unknown,
}

#[derive(Debug, Serialize)]
pub(super) struct Gate {
    pub(super) code: &'static str,
    pub(super) state: State,
    pub(super) detail: &'static str,
}

impl Gate {
    pub(super) fn new(code: &'static str, state: State, detail: &'static str) -> Self {
        Self {
            code,
            state,
            detail,
        }
    }
}

pub(super) fn demand_gate(listing: &Listing, market: &Market) -> Gate {
    let estimate = |value| value >= market.minimum_orders;
    match listing.demand {
        [None, None] => Gate::new(
            "LOW_DEMAND",
            State::Unknown,
            "нет оценок спроса, уточнить ожидаемые заказы",
        ),
        [Some(a), Some(b)] if estimate(a) != estimate(b) => Gate::new(
            "LOW_DEMAND",
            State::Unknown,
            "оценки по разные стороны порога, уточнить ожидаемые заказы",
        ),
        [Some(a), None] | [None, Some(a)] | [Some(a), Some(_)] if !estimate(a) => {
            Gate::new("LOW_DEMAND", State::Fail, "ожидаемые заказы ниже порога")
        }
        [Some(a), Some(b)]
            if market.demand_guard.as_ref().is_some_and(|guard| {
                a.min(b)
                    .checked_mul(guard.max_source_ratio)
                    .is_some_and(|limit| a.max(b) > limit)
            }) =>
        {
            Gate::new(
                "LOW_DEMAND",
                State::Unknown,
                "оценки спроса сильно расходятся, сверить с аналитиком",
            )
        }
        [Some(value), None] | [None, Some(value)]
            if market.demand_guard.as_ref().is_some_and(|guard| {
                listing
                    .reviews
                    .is_some_and(|reviews| reviews < guard.minimum_reviews)
                    && market
                        .minimum_orders
                        .checked_mul(guard.single_source_headroom)
                        .is_some_and(|limit| value < limit)
            }) =>
        {
            Gate::new(
                "LOW_DEMAND",
                State::Unknown,
                "единственная оценка близка к порогу при малом числе отзывов, сверить прогноз",
            )
        }
        _ => Gate::new(
            "LOW_DEMAND",
            State::Pass,
            "ожидаемые заказы достигают порога",
        ),
    }
}
