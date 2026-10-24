//! A reference exchange for the sync tests: it owns the true book, assigns Binance-style update
//! ids, and can hand out a snapshot of itself at any point. Generated streams from it are valid
//! by construction, so any resync the machine reports on them is a bug.

#![allow(dead_code)] // each test binary uses a different subset

use std::collections::BTreeMap;

use engine::market::book::Book;
use engine::market::sync::{DepthUpdate, Level, Snapshot};
use proptest::prelude::*;

/// One change to the reference book. A positive quantity sets the level, zero deletes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Bid { price: i64, qty: i64 },
    Ask { price: i64, qty: i64 },
}

/// The reference exchange.
#[derive(Debug, Clone)]
pub struct Exchange {
    bids: BTreeMap<i64, i64>,
    asks: BTreeMap<i64, i64>,
    last_id: u64,
}

impl Exchange {
    /// Both sides seeded so the first events already have a populated book to cross against.
    pub fn seeded(last_id: u64) -> Self {
        Self {
            bids: (95..=100).map(|p| (p, 1)).collect(),
            asks: (101..=106).map(|p| (p, 1)).collect(),
            last_id,
        }
    }

    pub fn last_id(&self) -> u64 {
        self.last_id
    }

    /// The current book labelled with `last_update_id = id`.
    pub fn snapshot_at(&self, id: u64) -> Snapshot {
        let levels = |m: &BTreeMap<i64, i64>| {
            m.iter()
                .map(|(&price, &qty)| Level { price, qty })
                .collect::<Vec<_>>()
        };
        Snapshot {
            last_update_id: id,
            bids: levels(&self.bids),
            asks: levels(&self.asks),
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        self.snapshot_at(self.last_id)
    }

    /// The current book as a [`Book`], for equality against the machine's.
    pub fn book(&self) -> Book {
        Book::from_snapshot(&self.snapshot())
    }

    /// Applies `ops` as one depth event spanning `span + 1` update ids. The book stays
    /// uncrossed: a bid at or above an ask removes those asks (and vice versa), and every
    /// touched price is reported with its final quantity, as Binance does.
    pub fn step(&mut self, ops: &[Op], span: u64) -> DepthUpdate {
        let prev_last_id = self.last_id;
        let first_id = prev_last_id + 1;
        let last_id = first_id + span;
        self.last_id = last_id;

        let mut touched_bids = BTreeMap::new();
        let mut touched_asks = BTreeMap::new();
        for op in ops {
            match *op {
                Op::Bid { price, qty } => {
                    set(&mut self.bids, price, qty);
                    touched_bids.insert(price, qty.max(0));
                    if qty > 0 {
                        let crossed: Vec<i64> =
                            self.asks.range(..=price).map(|(p, _)| *p).collect();
                        for p in crossed {
                            self.asks.remove(&p);
                            touched_asks.insert(p, 0);
                        }
                    }
                }
                Op::Ask { price, qty } => {
                    set(&mut self.asks, price, qty);
                    touched_asks.insert(price, qty.max(0));
                    if qty > 0 {
                        let crossed: Vec<i64> = self.bids.range(price..).map(|(p, _)| *p).collect();
                        for p in crossed {
                            self.bids.remove(&p);
                            touched_bids.insert(p, 0);
                        }
                    }
                }
            }
        }
        DepthUpdate {
            first_id,
            last_id,
            prev_last_id,
            bids: touched_bids
                .into_iter()
                .map(|(price, qty)| Level { price, qty })
                .collect(),
            asks: touched_asks
                .into_iter()
                .map(|(price, qty)| Level { price, qty })
                .collect(),
        }
    }
}

fn set(side: &mut BTreeMap<i64, i64>, price: i64, qty: i64) {
    if qty <= 0 {
        side.remove(&price);
    } else {
        side.insert(price, qty);
    }
}

/// A valid stream plus a snapshot of the exchange after each event.
#[derive(Debug, Clone)]
pub struct Scenario {
    pub events: Vec<DepthUpdate>,
    /// `snapshots[i]` is the book after `events[i]`, labelled with an id inside
    /// `[events[i].first_id, events[i].last_id]`, which is where a real snapshot can land.
    pub snapshots: Vec<Snapshot>,
    pub final_book: Book,
}

impl Scenario {
    pub fn len(&self) -> usize {
        self.events.len()
    }
}

pub fn op() -> impl Strategy<Value = Op> {
    let bid = (90i64..=110, 0i64..=5).prop_map(|(price, qty)| Op::Bid { price, qty });
    let ask = (90i64..=110, 0i64..=5).prop_map(|(price, qty)| Op::Ask { price, qty });
    prop_oneof![bid, ask]
}

/// `(ops, span, offset)` per event; `offset` picks where inside the id range a snapshot lands.
fn raw_event() -> impl Strategy<Value = (Vec<Op>, u64, u64)> {
    (prop::collection::vec(op(), 1..=4), 0u64..=3, 0u64..=3)
}

pub fn scenario(min_events: usize, max_events: usize) -> impl Strategy<Value = Scenario> {
    prop::collection::vec(raw_event(), min_events..=max_events).prop_map(|raw| {
        let mut exchange = Exchange::seeded(1_000);
        let mut events = Vec::with_capacity(raw.len());
        let mut snapshots = Vec::with_capacity(raw.len());
        for (ops, span, offset) in raw {
            let event = exchange.step(&ops, span);
            let id = event.first_id + offset % (span + 1);
            snapshots.push(exchange.snapshot_at(id));
            events.push(event);
        }
        Scenario {
            events,
            snapshots,
            final_book: exchange.book(),
        }
    })
}

/// A stream with a valid id chain but arbitrary levels, which may well cross.
pub fn wild_stream(max_events: usize) -> impl Strategy<Value = Vec<DepthUpdate>> {
    let level = (90i64..=110, 0i64..=5).prop_map(|(price, qty)| Level { price, qty });
    let levels = prop::collection::vec(level, 0..=3);
    prop::collection::vec((levels.clone(), levels, 0u64..=3), 1..=max_events).prop_map(|raw| {
        let mut last_id = 1_000;
        raw.into_iter()
            .map(|(bids, asks, span)| {
                let first_id = last_id + 1;
                let event = DepthUpdate {
                    first_id,
                    last_id: first_id + span,
                    prev_last_id: last_id,
                    bids,
                    asks,
                };
                last_id = event.last_id;
                event
            })
            .collect()
    })
}

/// Every level on both sides has a positive quantity.
pub fn all_levels_positive(book: &Book) -> bool {
    book.bids().chain(book.asks()).all(|l| l.qty > 0)
}
