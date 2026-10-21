//! The local order book: price ticks to lot units on each side, in integer arithmetic.

use std::collections::BTreeMap;

use super::sync::{DepthUpdate, Level, Snapshot};

/// A bid or ask side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Bid,
    Ask,
}

/// Price levels on both sides. Every level held has `qty > 0`; a zero quantity deletes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Book {
    bids: BTreeMap<i64, i64>,
    asks: BTreeMap<i64, i64>,
}

impl Book {
    /// Builds a book from a snapshot.
    #[must_use]
    pub fn from_snapshot(snapshot: &Snapshot) -> Self {
        let mut book = Self::default();
        for level in &snapshot.bids {
            book.set(Side::Bid, level.price, level.qty);
        }
        for level in &snapshot.asks {
            book.set(Side::Ask, level.price, level.qty);
        }
        book
    }

    /// Sets a level's absolute quantity; `qty <= 0` removes it (removing an absent level is
    /// normal on Binance's diff stream).
    pub fn set(&mut self, side: Side, price: i64, qty: i64) {
        let levels = match side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        };
        if qty <= 0 {
            levels.remove(&price);
        } else {
            levels.insert(price, qty);
        }
    }

    /// Applies every level in a depth update.
    pub fn apply(&mut self, update: &DepthUpdate) {
        for level in &update.bids {
            self.set(Side::Bid, level.price, level.qty);
        }
        for level in &update.asks {
            self.set(Side::Ask, level.price, level.qty);
        }
    }

    /// Highest bid as `(price, qty)`.
    #[must_use]
    pub fn best_bid(&self) -> Option<Level> {
        self.bids.iter().next_back().map(Level::from_entry)
    }

    /// Lowest ask as `(price, qty)`.
    #[must_use]
    pub fn best_ask(&self) -> Option<Level> {
        self.asks.iter().next().map(Level::from_entry)
    }

    /// True when both sides are populated and the best bid is at or above the best ask.
    /// Reported, never asserted: a cross is a signal for the caller to log or resync on.
    #[must_use]
    pub fn is_crossed(&self) -> bool {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => bid.price >= ask.price,
            _ => false,
        }
    }

    /// Number of bid and ask levels.
    #[must_use]
    pub fn depth(&self) -> (usize, usize) {
        (self.bids.len(), self.asks.len())
    }

    /// Bids from highest price down.
    pub fn bids(&self) -> impl Iterator<Item = Level> + '_ {
        self.bids.iter().rev().map(Level::from_entry)
    }

    /// Asks from lowest price up.
    pub fn asks(&self) -> impl Iterator<Item = Level> + '_ {
        self.asks.iter().map(Level::from_entry)
    }
}

impl Level {
    fn from_entry((price, qty): (&i64, &i64)) -> Self {
        Self {
            price: *price,
            qty: *qty,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn level(price: i64, qty: i64) -> Level {
        Level { price, qty }
    }

    #[test]
    fn snapshot_then_updates_keep_best_levels_right() {
        let snapshot = Snapshot {
            last_update_id: 1,
            bids: vec![level(100, 5), level(99, 1)],
            asks: vec![level(101, 2), level(103, 4)],
        };
        let mut book = Book::from_snapshot(&snapshot);
        assert_eq!(book.best_bid(), Some(level(100, 5)));
        assert_eq!(book.best_ask(), Some(level(101, 2)));
        assert!(!book.is_crossed());

        book.apply(&DepthUpdate {
            first_id: 2,
            last_id: 2,
            prev_last_id: 1,
            bids: vec![level(100, 0), level(98, 7)],
            asks: vec![level(101, 9)],
        });
        assert_eq!(book.best_bid(), Some(level(99, 1)));
        assert_eq!(book.best_ask(), Some(level(101, 9)));
        assert_eq!(book.depth(), (2, 2));
        assert_eq!(
            book.bids().collect::<Vec<_>>(),
            vec![level(99, 1), level(98, 7)]
        );
    }

    #[test]
    fn deleting_an_absent_level_is_a_no_op() {
        let mut book = Book::default();
        book.set(Side::Ask, 50, 0);
        assert_eq!(book.depth(), (0, 0));
        assert_eq!(book.best_ask(), None);
    }

    #[test]
    fn a_cross_is_reported_not_panicked() {
        let mut book = Book::default();
        book.set(Side::Bid, 101, 1);
        assert!(!book.is_crossed());
        book.set(Side::Ask, 100, 1);
        assert!(book.is_crossed());
        book.set(Side::Ask, 100, 0);
        book.set(Side::Ask, 102, 1);
        assert!(!book.is_crossed());
    }
}
