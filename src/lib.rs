#![deny(missing_docs)]
#![cfg_attr(test, feature(btree_cursors, assert_matches))]

//! This crate implements a [`BTreeMap`] similar to [`std::collections::BTreeMap`].
//!
//! Most of the implementation is in the [gb] module, see [`gb::BTreeMap`].
//!
//! One difference is the walk and `walk_mut` methods, which can be slightly more efficient than using range and `range_mut`.
//!
//! # ToDo
//!
//! More efficient implementation of append?
//!
//! Check memory used (std versus exp) ( Done: feature cap )
//!
//! Investigate why exp is using slightly more memory than std... with larger DB, exp wins.
//!
//! Maybe it is size of NonLeaf / Tree enum ( which could be reduced ).
//!
//! Either use Box ( which works to some extent ), or have new "NonLeafVec" type which stores only one length.
//!
//! # Example
//!
//! ```
//!     use btree_experiment::BTreeMap;
//!     let mut mymap = BTreeMap::new();
//!     mymap.insert("England", "London");
//!     mymap.insert("France", "Paris");
//!     println!("The capital of France is {}", mymap["France"]);
//! ```
//!
//!# Features
//!
//! This crate supports the following cargo features:
//! - `serde` : enables serialisation of [`BTreeMap`] via serde crate.
//! - `unsafe-optim` : uses unsafe code for extra optimisation.

/* Design of Leaf and NonLeaf storage...

   For a Leaf, the length could be stored in the parent...
   The capacity is determined from B.
   The storage is (aligned) array of keys + (aligned) array of values.

   For a NonLeaf, again the length can be stored in the parent...
   The storage is
       (aligned) array of keys
     + (aligned) array of values +
     + array of child pointers
     + array of child lengths (bytes).
   The key and value arrays are capacity B.
   The child length and child pointer arrays are capacity B + 1.
   B should be a number of form n*8 - 1, like 31 so B + 1 = 32, 32 bytes = 4 words.

   Node type can be determined by depth in the tree, the root BTreeMap can stored the overall tree depth.

   A "node reference" (not stored permanently) will consist of
       * a raw pointer to the node data
       * a raw pointer to the length byte in the parent node
*/

///  Not yet live... experiment storing vec lengths in parent.
pub mod lessmem;

/// Module with version of `BTreeMap` that allows B to be specified as generic constant.
pub mod gb;

mod vecs;

// Types for compatibility.

pub use gb::{Entry::Occupied, Entry::Vacant, UnorderedKeyError};

/// Default B value ( this is capacity, usually B is defined as B/2 + 1 ).
pub const DB: usize = 51;

/// `BTreeMap` similar to [`std::collections::BTreeMap`] with default node capacity [DB].
pub type BTreeMap<K, V> = gb::BTreeMap<K, V, DB>;

/// Entry in `BTreeMap`, returned by [`BTreeMap::entry`].
pub type Entry<'a, K, V> = gb::Entry<'a, K, V, DB>;

/// Consuming iterator returned by [`BTreeMap::into_iter`].
pub type IntoIter<K, V> = gb::IntoIter<K, V, DB>;

/// Consuming iterator returned by [`BTreeMap::into_keys`].
pub type IntoKeys<K, V> = gb::IntoKeys<K, V, DB>;

/// Consuming iterator returned by [`BTreeMap::into_values`].
pub type IntoValues<K, V> = gb::IntoValues<K, V, DB>;

/// Iterator returned by [`BTreeMap::iter_mut`].
pub type IterMut<'a, K, V> = gb::IterMut<'a, K, V, DB>;

/// Iterator returned by [`BTreeMap::iter`].
pub type Iter<'a, K, V> = gb::Iter<'a, K, V, DB>;

/// Cursor returned by [`BTreeMap::lower_bound`], [`BTreeMap::upper_bound`].
pub type Cursor<'a, K, V> = gb::Cursor<'a, K, V, DB>;

/// Cursor returned by [`BTreeMap::lower_bound_mut`], [`BTreeMap::upper_bound_mut`].
pub type CursorMut<'a, K, V> = gb::CursorMut<'a, K, V, DB>;

/// Cursor returned by [`CursorMut::with_mutable_key`].
pub type CursorMutKey<'a, K, V> = gb::CursorMutKey<'a, K, V, DB>;

/// Iterator returned by [`BTreeMap::extract_if`].
pub type ExtractIf<'a, K, V, F> = gb::ExtractIf<'a, K, V, DB, F>;

/// Iterator returned by [`BTreeMap::keys`].
pub type Keys<'a, K, V> = gb::Keys<'a, K, V, DB>;

/// Iterator returned by [`BTreeMap::values`].
pub type Values<'a, K, V> = gb::Values<'a, K, V, DB>;

/// Iterator returned by [`BTreeMap::range`].
pub type Range<'a, K, V> = gb::Range<'a, K, V, DB>;

/// Iterator returned by [`BTreeMap::range_mut`].
pub type RangeMut<'a, K, V> = gb::RangeMut<'a, K, V, DB>;

/// Occupied [Entry].
pub type OccupiedEntry<'a, K, V> = gb::OccupiedEntry<'a, K, V, DB>;

/// Vacant [Entry].
pub type VacantEntry<'a, K, V> = gb::VacantEntry<'a, K, V, DB>;

/// Error returned by [`BTreeMap::try_insert`].
pub type OccupiedError<'a, K, V> = gb::OccupiedError<'a, K, V, DB>;

// Tests.

#[cfg(all(test, not(miri), feature = "cap"))]
use {cap::Cap, std::alloc};

#[cfg(all(test, not(miri), feature = "cap"))]
#[global_allocator]
static ALLOCATOR: Cap<alloc::System> = Cap::new(alloc::System, usize::max_value());

#[cfg(test)]
fn print_memory() {
    #[cfg(all(test, not(miri), feature = "cap"))]
    println!("Memory allocated: {} bytes", ALLOCATOR.allocated());
}

/* mimalloc cannot be used with miri */
/*
#[cfg(all(test, not(miri), not(feature = "cap")))]
use mimalloc::MiMalloc;

#[cfg(all(test, not(miri), not(feature = "cap")))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
*/

#[cfg(test)]
mod mytests;

//#[cfg(test)] mod stdtests; // Increases compile/link time to 9 seconds from 3 seconds, so sometimes commented out!
