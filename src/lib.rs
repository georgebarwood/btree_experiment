//! This crate implements a BTreeMap similar to [std::collections::BTreeMap].
//!
//! One difference is the walk and walk_mut methods, which can be slightly more efficient than using range and range_mut.

// Note: some (crate) private methods of FixedCapVec are techically unsafe in release mode
// when the unsafe_optim feature is enabled, but are not declared as such to avoid littering
// the code with unsafe blocks.

#![deny(missing_docs)]
#![cfg_attr(test, feature(btree_cursors, assert_matches))]

/// Module with version of BTreeMap that allows B to specified as generic constant.
pub mod generic;
mod vecs;

// Types for compatibility.

pub use generic::{Entry::Occupied, Entry::Vacant, UnorderedKeyError};

/// Default node capacity = 41( B is usually defined as half this number ).
pub const DB: usize = 41;

/// BTreeMap similar to [std::collections::BTreeMap].
pub type BTreeMap<K, V> = generic::BTreeMap<K, V, DB>;

/// Entry in BTreeMap, returned by [BTreeMap::entry].
pub type Entry<'a, K, V> = generic::Entry<'a, K, V, DB>;

/// Consuming iterator returned by [BTreeMap::into_iter].
pub type IntoIter<K, V> = generic::IntoIter<K, V, DB>;

/// Consuming iterator returned by [BTreeMap::into_keys].
pub type IntoKeys<K, V> = generic::IntoKeys<K, V, DB>;

/// Consuming iterator returned by [BTreeMap::into_values].
pub type IntoValues<K, V> = generic::IntoValues<K, V, DB>;

/// Iterator returned by [BTreeMap::iter_mut].
pub type IterMut<'a, K, V> = generic::IterMut<'a, K, V, DB>;

/// Iterator returned by [BTreeMap::iter].
pub type Iter<'a, K, V> = generic::Iter<'a, K, V, DB>;

/// Cursor returned by [BTreeMap::lower_bound], [BTreeMap::upper_bound].
pub type Cursor<'a, K, V> = generic::Cursor<'a, K, V, DB>;

/// Cursor returned by [BTreeMap::lower_bound_mut], [BTreeMap::upper_bound_mut].
pub type CursorMut<'a, K, V> = generic::CursorMut<'a, K, V, DB>;

/// Cursor returned by [CursorMut::with_mutable_key].
pub type CursorMutKey<'a, K, V> = generic::CursorMutKey<'a, K, V, DB>;

/// Iterator returned by [BTreeMap::extract_if].
pub type ExtractIf<'a, K, V, F> = generic::ExtractIf<'a, K, V, DB, F>;

/// Iterator returned by [BTreeMap::keys].
pub type Keys<'a, K, V> = generic::Keys<'a, K, V, DB>;

/// Iterator returned by [BTreeMap::values].
pub type Values<'a, K, V> = generic::Values<'a, K, V, DB>;

/// Iterator returned by [BTreeMap::range].
pub type Range<'a, K, V> = generic::Range<'a, K, V, DB>;

/// Iterator returned by [BTreeMap::range_mut].
pub type RangeMut<'a, K, V> = generic::RangeMut<'a, K, V, DB>;

/// Occupied [Entry].
pub type OccupiedEntry<'a, K, V> = generic::OccupiedEntry<'a, K, V, DB>;

/// Vacant [Entry].
pub type VacantEntry<'a, K, V> = generic::VacantEntry<'a, K, V, DB>;

/// Error returned by [BTreeMap::try_insert].
pub type OccupiedError<'a, K, V> = generic::OccupiedError<'a, K, V, DB>;



// Tests.

/* mimalloc cannot be used with miri */
#[cfg(all(test, not(miri)))]
use mimalloc::MiMalloc;

#[cfg(all(test, not(miri)))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[cfg(test)]
mod mytests;

#[cfg(test)]
mod stdtests; // Increases compile/link time to 9 seconds from 3 seconds, so sometimes commented out!

