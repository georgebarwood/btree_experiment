#![deny(missing_docs)]
#![cfg_attr(test, feature(btree_cursors, assert_matches))]

//! This crate implements a [`BTreeMap`] similar to [`std::collections::BTreeMap`].
//!
//! The standard BtreeMap can use up to twice as much memory as required, this BTreeMap
//! only allocates what is needed ( or a little more to avoid allocating too often ), so
//! memory use can be up to 50% less.
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

use std::{
    borrow::Borrow,
    cmp::Ordering,
    error::Error,
    fmt,
    fmt::Debug,
    iter::FusedIterator,
    marker::PhantomData,
    mem,
    ops::{Bound, RangeBounds},
};

/// `BTreeMap` similar to [`std::collections::BTreeMap`].
///
/// General guide to implementation:
///
/// [`BTreeMap`] has a length and a `Tree`, where `Tree` is an enum that can be `Leaf` or `NonLeaf`.
///
/// The [Entry] API is implemented using [`CursorMut`].
///
/// [`CursorMut`] is implemented using [`CursorMutKey`] which has a stack of raw pointer/index pairs
/// to keep track of non-leaf positions.
///
/// Roughly speaking, unsafe code is limited to the vecs module and the implementation of [`CursorMut`] and [`CursorMutKey`].

#[derive(Clone)]
pub struct BTreeMap<K, V, A: AllocTuning = DefaultAllocTuning> {
    len: usize,
    tree: Tree<K, V>,
    atune: A,
}
impl<K, V> Default for BTreeMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> BTreeMap<K, V> {
    /// Returns a new, empty map.
    #[must_use]
    pub fn new() -> Self {
        Self::with_tuning(DefaultAllocTuning::default())
    }
}

impl<K, V, A: AllocTuning> BTreeMap<K, V, A> {
    #[cfg(test)]
    pub(crate) fn check(&self) {}

    /// Returns a new, empty map with specified allocation tuning.
    ///
    /// # Example
    ///
    /// ```
    ///     use btree_experiment::{BTreeMap,DefaultAllocTuning};
    ///     let mut mymap = BTreeMap::with_tuning(DefaultAllocTuning::new(8,2));
    ///     mymap.insert("England", "London");
    ///     mymap.insert("France", "Paris");
    ///     println!("The capital of France is {}", mymap["France"]);
    /// ```
    #[must_use]
    pub fn with_tuning(atune: A) -> Self {
        Self {
            len: 0,
            tree: Tree::new(),
            atune,
        }
    }

    /// Get a cloned copy of the tuning.
    pub fn get_tuning(&self) -> A {
        self.atune.clone()
    }

    /// Update the allocation tuning for the map.
    pub fn set_tuning(&mut self, atune: A) -> A {
        mem::replace(&mut self.atune, atune)
    }

    /// Clear the map.
    pub fn clear(&mut self) {
        self.len = 0;
        self.tree = Tree::new();
    }

    /// Get number of key-value pairs in the map.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Is the map empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get Entry for map key.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, A>
    where
        K: Ord,
    {
        let cursor = self.lower_bound_mut(Bound::Included(&key));
        let found = if let Some(kv) = cursor.peek_next() {
            kv.0 == &key
        } else {
            false
        };
        if found {
            Entry::Occupied(OccupiedEntry { cursor })
        } else {
            Entry::Vacant(VacantEntry { key, cursor })
        }
    }

    /// Get first Entry.
    pub fn first_entry(&mut self) -> Option<OccupiedEntry<'_, K, V, A>>
    where
        K: Ord,
    {
        if self.is_empty() {
            None
        } else {
            let cursor = self.lower_bound_mut(Bound::Unbounded);
            Some(OccupiedEntry { cursor })
        }
    }

    /// Get last Entry.
    pub fn last_entry(&mut self) -> Option<OccupiedEntry<'_, K, V, A>>
    where
        K: Ord,
    {
        if self.is_empty() {
            None
        } else {
            let mut cursor = self.upper_bound_mut(Bound::Unbounded);
            cursor.prev();
            Some(OccupiedEntry { cursor })
        }
    }

    /// Insert key-value pair into map, or if key is already in map, replaces value and returns old value.
    pub fn insert(&mut self, key: K, value: V) -> Option<V>
    where
        K: Ord,
    {
        let mut x = InsertCtx {
            value: Some(value),
            split: None,
            atune: &self.atune,
        };
        self.tree.insert(key, &mut x);
        if let Some(split) = x.split {
            self.tree.new_root(split);
        }
        if x.value.is_none() {
            self.len += 1;
        }
        x.value
    }

    /// Tries to insert a key-value pair into the map, and returns
    /// a mutable reference to the value in the entry.
    ///
    /// If the map already had this key present, nothing is updated, and
    /// an error containing the occupied entry and the value is returned.
    pub fn try_insert(&mut self, key: K, value: V) -> Result<&mut V, OccupiedError<'_, K, V, A>>
    where
        K: Ord,
    {
        match self.entry(key) {
            Entry::Occupied(entry) => Err(OccupiedError { entry, value }),
            Entry::Vacant(entry) => Ok(entry.insert(value)),
        }
    }

    /// Does the map have an entry for the specified key.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.get(key).is_some()
    }

    /// Remove key-value pair from map, returning just the value.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.remove_entry(key).map(|(_k, v)| v)
    }

    /// Remove key-value pair from map.
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let result = self.tree.remove(key, &self.atune)?;
        self.len -= 1;
        Some(result)
    }

    /// Remove first key-value pair from map.
    pub fn pop_first(&mut self) -> Option<(K, V)> {
        let result = self.tree.pop_first(&self.atune)?;
        self.len -= 1;
        Some(result)
    }

    /// Remove last key-value pair from map.
    pub fn pop_last(&mut self) -> Option<(K, V)> {
        let result = self.tree.pop_last(&self.atune)?;
        self.len -= 1;
        Some(result)
    }

    /// Remove all key-value pairs, visited in ascending order, for which f returns false.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
        K: Ord,
    {
        let mut c = self.lower_bound_mut(Bound::Unbounded);
        while let Some((k, v)) = c.next() {
            if !f(k, v) {
                c.remove_prev();
            }
        }
    }

    /// Get reference to the value corresponding to the key.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.tree.get(key)
    }

    /// Get a mutable reference to the value corresponding to the key.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.tree.get_mut(key)
    }

    /// Get references to the corresponding key and value.
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.tree.get_key_value(key)
    }

    /// Get references to first key and value.
    #[must_use]
    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        self.tree.iter().next()
    }

    /// Gets references to last key and value.
    #[must_use]
    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        self.tree.iter().next_back()
    }

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    ///
    /// If a key from `other` is already present in `self`, the respective
    /// value from `self` will be overwritten with the respective value from `other`.
    pub fn append(&mut self, other: &mut BTreeMap<K, V, A>)
    where
        K: Ord,
    {
        let rep = Tree::new();
        let tree = mem::replace(&mut other.tree, rep);
        let temp = BTreeMap {
            len: other.len,
            tree,
            atune: other.atune.clone(),
        };
        other.len = 0;
        for (k, v) in temp {
            self.insert(k, v);
        }
    }

    /// Splits the collection into two at the given key.
    /// Returns everything after the given key, including the key.
    pub fn split_off<Q: ?Sized + Ord>(&mut self, key: &Q) -> Self
    where
        K: Borrow<Q> + Ord,
        A: Default,
    {
        let mut map = Self::with_tuning(self.atune.clone());
        let mut from = self.lower_bound_mut(Bound::Included(key));
        let mut to = map.lower_bound_mut(Bound::Unbounded);
        while let Some((k, v)) = from.remove_next() {
            unsafe {
                to.insert_before_unchecked(k, v);
            }
        }
        map
    }

    /// Returns iterator that visits all elements (key-value pairs) in ascending key order
    /// and uses a closure to determine if an element should be removed.
    pub fn extract_if<F>(&mut self, pred: F) -> ExtractIf<'_, K, V, A, F>
    where
        K: Ord,
        F: FnMut(&K, &mut V) -> bool,
    {
        let source = self.lower_bound_mut(Bound::Unbounded);
        ExtractIf { source, pred }
    }

    /// Get iterator of references to key-value pairs.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            len: self.len,
            inner: self.tree.iter(),
        }
    }

    /// Get iterator of mutable references to key-value pairs.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            len: self.len,
            inner: self.tree.iter_mut(),
        }
    }

    /// Get iterator for range of references to key-value pairs.
    pub fn range<T, R>(&self, range: R) -> Range<'_, K, V>
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        check_range(&range);
        self.tree.range(&range)
    }

    /// Get iterator for range of mutable references to key-value pairs.
    /// A key can be mutated, provided it does not change the map order.
    pub fn range_mut<T, R>(&mut self, range: R) -> RangeMut<'_, K, V>
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        check_range(&range);
        self.tree.range_mut(&range)
    }

    /// Get iterator of references to keys.
    #[must_use]
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys(self.iter())
    }

    /// Get iterator of references to values.
    #[must_use]
    pub fn values(&self) -> Values<'_, K, V> {
        Values(self.iter())
    }

    /// Get iterator of mutable references to values.
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        ValuesMut(self.iter_mut())
    }

    /// Get consuming iterator that returns all the keys, in sorted order.
    #[must_use]
    pub fn into_keys(self) -> IntoKeys<K, V> {
        IntoKeys(self.into_iter())
    }

    /// Get consuming iterator that returns all the values, in sorted order.
    #[must_use]
    pub fn into_values(self) -> IntoValues<K, V> {
        IntoValues(self.into_iter())
    }

    /// Get cursor positioned just after bound.
    #[must_use]
    pub fn lower_bound<Q>(&self, bound: Bound<&Q>) -> Cursor<'_, K, V, A>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        Cursor::lower_bound(self, bound)
    }

    /// Get cursor positioned just before bound.
    #[must_use]
    pub fn upper_bound<Q>(&self, bound: Bound<&Q>) -> Cursor<'_, K, V, A>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        Cursor::upper_bound(self, bound)
    }

    /// Get a cursor positioned just after bound that permits map mutation.
    pub fn lower_bound_mut<Q>(&mut self, bound: Bound<&Q>) -> CursorMut<'_, K, V, A>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        CursorMut::lower_bound(self, bound)
    }

    /// Get a cursor positioned just before bound that permits map mutation.
    pub fn upper_bound_mut<Q>(&mut self, bound: Bound<&Q>) -> CursorMut<'_, K, V, A>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        CursorMut::upper_bound(self, bound)
    }
} // End impl BTreeMap

use std::hash::{Hash, Hasher};
impl<K: Hash, V: Hash> Hash for BTreeMap<K, V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // state.write_length_prefix(self.len());
        for elt in self {
            elt.hash(state);
        }
    }
}
impl<K: PartialEq, V: PartialEq> PartialEq for BTreeMap<K, V> {
    fn eq(&self, other: &BTreeMap<K, V>) -> bool {
        self.len() == other.len() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}
impl<K: Eq, V: Eq> Eq for BTreeMap<K, V> {}

impl<K: PartialOrd, V: PartialOrd> PartialOrd for BTreeMap<K, V> {
    fn partial_cmp(&self, other: &BTreeMap<K, V>) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}
impl<K: Ord, V: Ord> Ord for BTreeMap<K, V> {
    fn cmp(&self, other: &BTreeMap<K, V>) -> Ordering {
        self.iter().cmp(other.iter())
    }
}
impl<K, V, A: AllocTuning> IntoIterator for BTreeMap<K, V, A> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    /// Convert `BTreeMap` to [`IntoIter`].
    fn into_iter(self) -> IntoIter<K, V> {
        IntoIter::new(self)
    }
}
impl<'a, K, V> IntoIterator for &'a BTreeMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;
    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}
impl<'a, K, V> IntoIterator for &'a mut BTreeMap<K, V> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;
    fn into_iter(self) -> IterMut<'a, K, V> {
        self.iter_mut()
    }
}
impl<K: Ord, V> FromIterator<(K, V)> for BTreeMap<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> BTreeMap<K, V> {
        let mut map = BTreeMap::new();
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}
impl<K, V, const N: usize> From<[(K, V); N]> for BTreeMap<K, V>
where
    K: Ord,
{
    fn from(arr: [(K, V); N]) -> BTreeMap<K, V> {
        let mut map = BTreeMap::new();
        for (k, v) in arr {
            map.insert(k, v);
        }
        map
    }
}
impl<K, V> Extend<(K, V)> for BTreeMap<K, V>
where
    K: Ord,
{
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = (K, V)>,
    {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}
impl<'a, K, V> Extend<(&'a K, &'a V)> for BTreeMap<K, V>
where
    K: Ord + Copy,
    V: Copy,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (&'a K, &'a V)>,
    {
        for (&k, &v) in iter {
            self.insert(k, v);
        }
    }
}
impl<K, Q, V> std::ops::Index<&Q> for BTreeMap<K, V>
where
    K: Borrow<Q> + Ord,
    Q: Ord + ?Sized,
{
    type Output = V;

    /// Returns a reference to the value corresponding to the supplied key.
    ///
    /// Panics if the key is not present in the `BTreeMap`.
    #[inline]
    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}
impl<K: Debug, V: Debug> Debug for BTreeMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

#[cfg(feature = "serde")]
use serde::{
    de::{MapAccess, Visitor},
    ser::SerializeMap,
    Deserialize, Deserializer, Serialize,
};

#[cfg(feature = "serde")]
impl<K, V> Serialize for BTreeMap<K, V>
where
    K: serde::Serialize,
    V: serde::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.len()))?;
        for (k, v) in self {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}

#[cfg(feature = "serde")]
struct BTreeMapVisitor<K, V> {
    marker: PhantomData<fn() -> BTreeMap<K, V>>,
}

#[cfg(feature = "serde")]
impl<K, V> BTreeMapVisitor<K, V> {
    fn new() -> Self {
        BTreeMapVisitor {
            marker: PhantomData,
        }
    }
}

#[cfg(feature = "serde")]
impl<'de, K, V> Visitor<'de> for BTreeMapVisitor<K, V>
where
    K: Deserialize<'de> + Ord,
    V: Deserialize<'de>,
{
    type Value = BTreeMap<K, V>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("BTreeMap")
    }

    fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut map = BTreeMap::new();
        let mut tuning = map.get_tuning();
        tuning.set_seq();
        let save = map.set_tuning(tuning);
        {
            let mut c = map.lower_bound_mut(Bound::Unbounded);
            loop {
                if let Some((k, v)) = access.next_entry()? {
                    if let Some((pk, _)) = c.peek_prev() {
                        if pk >= &k {
                            map.insert(k, v);
                            break;
                        }
                    }
                    unsafe {
                        c.insert_before_unchecked(k, v);
                    }
                } else {
                    return Ok(map);
                }
            }
        }
        map.set_tuning(save);
        while let Some((k, v)) = access.next_entry()? {
            map.insert(k, v);
        }
        return Ok(map);
    }
}

#[cfg(feature = "serde")]
impl<'de, K, V> Deserialize<'de> for BTreeMap<K, V>
where
    K: Deserialize<'de> + Ord,
    V: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(BTreeMapVisitor::new())
    }
}

/// Type returned by [AllocTuning::full_action].
pub enum FullAction {
    /// Vec is to be split at indicated point with the indicated new allocations.
    Split(usize, usize, usize),
    /// Vec is to be extended to indicated length.
    Extend(usize),
}

/// Trait for controlling storage allocation for [BTreeMap].
pub trait AllocTuning: Clone + Default {
    /// Determine what to do when the size of an underlying BTree vector needs to be increased.
    fn full_action(&self, i: usize, len: usize) -> FullAction;
    /// Returns the new allocation if the allocation should be reduced based on the current length and allocation.
    fn space_action(&self, state: (usize, usize)) -> Option<usize>;
    /// Set allocation mode to be optimal for sequential (ordered) inserts.
    fn set_seq(&mut self);
}

/// Default implementation of [AllocTuning]. Default branch is 64, default allocation unit is 8.
#[derive(Clone)]
pub struct DefaultAllocTuning {
    branch: u16,
    alloc_unit: u8,
}
impl Default for DefaultAllocTuning {
    fn default() -> Self {
        Self {
            branch: 64,
            alloc_unit: 8,
        }
    }
}
impl AllocTuning for DefaultAllocTuning {
    fn full_action(&self, i: usize, len: usize) -> FullAction {
        let lim = (self.branch as usize) * 2 + 1;
        if len >= lim {
            let b = len / 2;
            let r = usize::from(i > b);
            let au = self.alloc_unit as usize;
            FullAction::Split(b, b + (1 - r) * au, (len - b - 1) + r * au)
        } else {
            let mut na = len + self.alloc_unit as usize;
            if na > lim {
                na = lim;
            }
            FullAction::Extend(na)
        }
    }
    fn space_action(&self, (len, alloc): (usize, usize)) -> Option<usize> {
        if alloc - len >= self.alloc_unit as usize {
            Some(len)
        } else {
            None
        }
    }
    fn set_seq(&mut self) {
        self.alloc_unit = u8::MAX;
    }
}
impl DefaultAllocTuning {
    /// Construct with specified branch and allocation unit.
    pub fn new(branch: u16, alloc_unit: u8) -> Self {
        assert!(branch >= 6);
        assert!(branch <= 512);
        assert!(alloc_unit > 0);
        Self { branch, alloc_unit }
    }
}

// Vector types.
type StkVec<T> = arrayvec::ArrayVec<T, 15>;

mod vecs;
use vecs::*;

type TreeVec<K, V> = ShortVec<Tree<K, V>>;

type Split<K, V> = ((K, V), Tree<K, V>);

struct InsertCtx<'a, K, V, A: AllocTuning> {
    value: Option<V>,
    split: Option<Split<K, V>>,
    atune: &'a A,
}

fn check_range<T, R>(range: &R)
where
    T: Ord + ?Sized,
    R: RangeBounds<T>,
{
    use Bound::{Excluded, Included};
    match (range.start_bound(), range.end_bound()) {
        (Included(s) | Excluded(s), Included(e)) | (Included(s), Excluded(e)) => {
            assert!(e >= s, "range start is greater than range end in BTreeMap");
        }
        (Excluded(s), Excluded(e)) => {
            assert!(
                e != s,
                "range start and end are equal and excluded in BTreeMap"
            );
            assert!(e >= s, "range start is greater than range end in BTreeMap");
        }
        _ => {}
    }
}

#[derive(Debug, Clone)]
enum Tree<K, V> {
    L(Leaf<K, V>),
    NL(NonLeaf<K, V>),
}
impl<K, V> Default for Tree<K, V> {
    fn default() -> Self {
        Self::new()
    }
}
impl<K, V> Tree<K, V> {
    fn new() -> Self {
        Tree::L(Leaf::new())
    }

    fn insert<A: AllocTuning>(&mut self, key: K, x: &mut InsertCtx<K, V, A>)
    where
        K: Ord,
    {
        match self {
            Tree::L(leaf) => leaf.insert(key, x),
            Tree::NL(nonleaf) => nonleaf.insert(key, x),
        }
    }

    fn new_root(&mut self, (med, right): Split<K, V>) {
        let mut nl = NonLeafInner::new();
        nl.v.0.set_alloc(1);
        nl.v.0.push(med);
        nl.c.set_alloc(2);
        nl.c.push(mem::take(self));
        nl.c.push(right);
        *self = Tree::NL(nl);
    }

    unsafe fn nonleaf(&mut self) -> &mut NonLeaf<K, V> {
        match self {
            Tree::NL(nl) => nl,
            Tree::L(_) => unsafe { std::hint::unreachable_unchecked() },
        }
    }

    unsafe fn leaf(&mut self) -> &mut Leaf<K, V> {
        match self {
            Tree::L(leaf) => leaf,
            Tree::NL(_) => unsafe { std::hint::unreachable_unchecked() },
        }
    }

    fn remove<Q, A: AllocTuning>(&mut self, key: &Q, atune: &A) -> Option<(K, V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self {
            Tree::L(leaf) => leaf.remove(key, atune),
            Tree::NL(nonleaf) => nonleaf.remove(key, atune),
        }
    }

    fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let mut s = self;
        loop {
            match s {
                Tree::L(leaf) => return leaf.get_key_value(key),
                Tree::NL(nl) => match nl.v.look(key) {
                    Ok(i) => return Some(nl.v.0.ix(i)),
                    Err(i) => s = nl.c.ix(i),
                },
            }
        }
    }

    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let mut s = self;
        loop {
            match s {
                Tree::L(leaf) => return leaf.get(key),
                Tree::NL(nl) => match nl.v.look(key) {
                    Ok(i) => return Some(nl.v.0.ixv(i)),
                    Err(i) => s = nl.c.ix(i),
                },
            }
        }
    }

    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let mut s = self;
        loop {
            match s {
                Tree::L(leaf) => return leaf.get_mut(key),
                Tree::NL(nl) => match nl.v.look(key) {
                    Ok(i) => return Some(nl.v.0.ixmv(i)),
                    Err(i) => s = nl.c.ixm(i),
                },
            }
        }
    }

    fn pop_first<A: AllocTuning>(&mut self, atune: &A) -> Option<(K, V)> {
        match self {
            Tree::L(leaf) => leaf.pop_first(atune),
            Tree::NL(nonleaf) => nonleaf.pop_first(atune),
        }
    }

    fn pop_last<A: AllocTuning>(&mut self, atune: &A) -> Option<(K, V)> {
        match self {
            Tree::L(leaf) => leaf.pop_last(atune),
            Tree::NL(nonleaf) => nonleaf.pop_last(atune),
        }
    }

    fn iter_mut(&mut self) -> RangeMut<'_, K, V> {
        let mut x = RangeMut::new();
        x.push_tree(self, true);
        x
    }

    fn iter(&self) -> Range<'_, K, V> {
        let mut x = Range::new();
        x.push_tree(self, true);
        x
    }

    fn range_mut<T, R>(&mut self, range: &R) -> RangeMut<'_, K, V>
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        let mut x = RangeMut::new();
        x.push_range(self, range, true);
        x
    }

    fn range<T, R>(&self, range: &R) -> Range<'_, K, V>
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        let mut x = Range::new();
        x.push_range(self, range, true);
        x
    }
} // End impl Tree

impl<K, V> Default for Leaf<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
struct Leaf<K, V>(PairVec<K, V>);

impl<K, V> Leaf<K, V> {
    fn new() -> Self {
        Self(PairVec::new())
    }

    fn full(&self) -> bool {
        self.0.full()
    }

    fn into_iter(mut self) -> IntoIterPairVec<K, V> {
        let v = mem::take(&mut self.0);
        v.into_iter()
    }

    fn look_to<Q>(&self, n: usize, key: &Q) -> Result<usize, usize>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.0.search_to(n, key)
    }

    #[inline]
    fn look<Q>(&self, key: &Q) -> Result<usize, usize>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.0.search(key)
    }

    fn skip<Q>(&self, key: &Q) -> usize
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self.0.search(key) {
            Ok(i) | Err(i) => i,
        }
    }

    fn skip_over<Q>(&self, key: &Q) -> usize
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self.0.search(key) {
            Ok(i) => i + 1,
            Err(i) => i,
        }
    }

    fn get_lower<Q>(&self, bound: Bound<&Q>) -> usize
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match bound {
            Bound::Unbounded => 0,
            Bound::Included(k) => self.skip(k),
            Bound::Excluded(k) => self.skip_over(k),
        }
    }

    fn get_upper<Q>(&self, bound: Bound<&Q>) -> usize
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match bound {
            Bound::Unbounded => self.0.len(),
            Bound::Included(k) => self.skip_over(k),
            Bound::Excluded(k) => self.skip(k),
        }
    }

    fn insert<A: AllocTuning>(&mut self, key: K, x: &mut InsertCtx<K, V, A>)
    where
        K: Ord,
    {
        let mut i = match self.look(&key) {
            Ok(i) => {
                let value = x.value.take().unwrap();
                let (k, v) = self.0.ixbm(i);
                *k = key;
                x.value = Some(mem::replace(v, value));
                return;
            }
            Err(i) => i,
        };
        let value = x.value.take().unwrap();
        if self.full() {
            match x.atune.full_action(i, self.0.len()) {
                FullAction::Split(b, a1, a2) => {
                    let (med, mut right) = self.0.split(b, a1, a2);
                    if i > b {
                        i -= b + 1;
                        assert!(i < a2);
                        right.insert(i, (key, value));
                    } else {
                        assert!(i < a1);
                        self.0.insert(i, (key, value));
                    }
                    let right = Tree::L(Self(right));
                    x.split = Some((med, right));
                    return;
                }
                FullAction::Extend(na) => self.0.set_alloc(na),
            }
        }
        self.0.insert(i, (key, value));
    }

    fn remove<Q, A: AllocTuning>(&mut self, key: &Q, atune: &A) -> Option<(K, V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let ix = self.look(key).ok()?;
        let result = self.0.remove(ix);
        if let Some(na) = atune.space_action(self.0.state()) {
            self.0.set_alloc(na);
        }
        Some(result)
    }

    #[inline]
    fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        Some(self.0.ix(self.look(key).ok()?))
    }

    #[inline]
    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        Some(self.0.ixv(self.look(key).ok()?))
    }

    #[inline]
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        Some(self.0.ixmv(self.look(key).ok()?))
    }

    fn pop_first<A: AllocTuning>(&mut self, atune: &A) -> Option<(K, V)> {
        if self.0.is_empty() {
            return None;
        }
        let result = Some(self.0.remove(0));
        if let Some(na) = atune.space_action(self.0.state()) {
            self.0.set_alloc(na);
        }
        result
    }

    fn pop_last<A: AllocTuning>(&mut self, atune: &A) -> Option<(K, V)> {
        if self.0.is_empty() {
            return None;
        }
        let result = self.0.pop();
        if let Some(na) = atune.space_action(self.0.state()) {
            self.0.set_alloc(na);
        }
        result
    }

    fn get_xy<T, R>(&self, range: &R) -> (usize, usize)
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        let y = match range.end_bound() {
            Bound::Unbounded => self.0.len(),
            Bound::Included(k) => self.skip_over(k),
            Bound::Excluded(k) => self.skip(k),
        };
        let x = match range.start_bound() {
            Bound::Unbounded => 0,
            Bound::Included(k) => match self.look_to(y, k) {
                Ok(i) => i,
                Err(i) => i,
            },
            Bound::Excluded(k) => match self.look_to(y, k) {
                Ok(i) => i + 1,
                Err(i) => i,
            },
        };
        (x, y)
    }

    fn iter_mut(&mut self) -> IterMutPairVec<'_, K, V> {
        self.0.iter_mut()
    }

    fn iter(&self) -> IterPairVec<'_, K, V> {
        self.0.iter()
    }
} // End impl Leaf

/* Boxing NonLeaf saves some memory by reducing size of Tree enum */
type NonLeaf<K, V> = Box<NonLeafInner<K, V>>;

#[derive(Debug, Clone)]
struct NonLeafInner<K, V> {
    v: Leaf<K, V>,
    c: TreeVec<K, V>,
}
impl<K, V> NonLeafInner<K, V> {
    fn new() -> Box<Self> {
        Box::new(Self {
            v: Leaf::new(),
            c: TreeVec::new(),
        })
    }

    #[allow(clippy::type_complexity)]
    fn into_iter(mut self) -> (IntoIterPairVec<K, V>, IntoIterShortVec<Tree<K, V>>) {
        let v = mem::take(&mut self.v);
        let c = mem::take(&mut self.c);
        (v.into_iter(), c.into_iter())
    }

    fn remove_at<A: AllocTuning>(&mut self, i: usize, atune: &A) -> ((K, V), usize) {
        if let Some(kv) = self.c.ixm(i).pop_last(atune) {
            let (kp, vp) = self.v.0.ixbm(i);
            let k = mem::replace(kp, kv.0);
            let v = mem::replace(vp, kv.1);
            ((k, v), i + 1)
        } else {
            self.c.remove(i);
            let kv = self.v.0.remove(i);
            if let Some(na) = atune.space_action(self.v.0.state()) {
                self.v.0.set_alloc(na);
                self.c.set_alloc(na + 1);
            }
            (kv, i)
        }
    }

    fn split(&mut self, b: usize, a1: usize, a2: usize) -> ((K, V), Box<Self>) {
        let (med, right) = self.v.0.split(b, a1, a2);
        let right = Box::new(Self {
            v: Leaf(right),
            c: self.c.split(b + 1, a1 + 1, a2 + 1),
        });
        debug_assert!(right.v.0.alloc + 1 == right.c.alloc);
        debug_assert!(self.v.0.alloc + 1 == self.c.alloc);
        (med, right)
    }

    fn insert<A: AllocTuning>(&mut self, key: K, x: &mut InsertCtx<K, V, A>)
    where
        K: Ord,
    {
        match self.v.look(&key) {
            Ok(i) => {
                let value = x.value.take().unwrap();
                let (kp, vp) = self.v.0.ixbm(i);
                *kp = key;
                x.value = Some(mem::replace(vp, value));
            }
            Err(mut i) => {
                self.c.ixm(i).insert(key, x);
                if let Some((med, right)) = x.split.take() {
                    if self.v.full() {
                        match x.atune.full_action(i, self.v.0.len()) {
                            FullAction::Split(b, a1, a2) => {
                                let (pmed, mut pright) = self.split(b, a1, a2);
                                if i > b {
                                    i -= b + 1;
                                    assert!(i < a2);
                                    pright.v.0.insert(i, med);
                                    pright.c.insert(i + 1, right);
                                } else {
                                    assert!(i < a1);
                                    self.v.0.insert(i, med);
                                    self.c.insert(i + 1, right);
                                }
                                x.split = Some((pmed, Tree::NL(pright)));
                                return;
                            }
                            FullAction::Extend(to) => {
                                self.v.0.set_alloc(to);
                                self.c.set_alloc(to + 1);
                            }
                        }
                    }
                    self.v.0.insert(i, med);
                    self.c.insert(i + 1, right);
                }
            }
        }
    }

    fn remove<Q, A: AllocTuning>(&mut self, key: &Q, atune: &A) -> Option<(K, V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self.v.look(key) {
            Ok(i) => Some(self.remove_at(i, atune).0),
            Err(i) => self.c.ixm(i).remove(key, atune),
        }
    }

    fn pop_first<A: AllocTuning>(&mut self, atune: &A) -> Option<(K, V)> {
        if let Some(x) = self.c.ixm(0).pop_first(atune) {
            Some(x)
        } else if self.v.0.is_empty() {
            None
        } else {
            self.c.remove(0);
            let result = Some(self.v.0.remove(0));
            if let Some(na) = atune.space_action(self.v.0.state()) {
                self.v.0.set_alloc(na);
                self.c.set_alloc(na + 1);
            }
            result
        }
    }

    fn pop_last<A: AllocTuning>(&mut self, atune: &A) -> Option<(K, V)> {
        let i = self.c.len();
        if let Some(x) = self.c.ixm(i - 1).pop_last(atune) {
            Some(x)
        } else if self.v.0.is_empty() {
            None
        } else {
            self.c.pop();
            let result = self.v.0.pop();
            if let Some(na) = atune.space_action(self.v.0.state()) {
                self.v.0.set_alloc(na);
                self.c.set_alloc(na + 1);
            }
            result
        }
    }
} // End impl NonLeafInner

/// Error returned by [`BTreeMap::try_insert`].
pub struct OccupiedError<'a, K, V, A: AllocTuning>
where
    K: 'a,
    V: 'a,
{
    /// Occupied entry, has the key that was not inserted.
    pub entry: OccupiedEntry<'a, K, V, A>,
    /// Value that was not inserted.
    pub value: V,
}
impl<K: Debug + Ord, V: Debug, A: AllocTuning> Debug for OccupiedError<'_, K, V, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OccupiedError")
            .field("key", self.entry.key())
            .field("old_value", self.entry.get())
            .field("new_value", &self.value)
            .finish()
    }
}
impl<'a, K: Debug + Ord, V: Debug, A: AllocTuning> fmt::Display for OccupiedError<'a, K, V, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "failed to insert {:?}, key {:?} already exists with value {:?}",
            self.value,
            self.entry.key(),
            self.entry.get(),
        )
    }
}
impl<'a, K: Debug + Ord, V: Debug, A: AllocTuning> Error for OccupiedError<'a, K, V, A> {}

/// Entry in `BTreeMap`, returned by [`BTreeMap::entry`].
pub enum Entry<'a, K, V, A: AllocTuning> {
    /// Vacant entry - map doesn't yet contain key.
    Vacant(VacantEntry<'a, K, V, A>),
    /// Occupied entry - map already contains key.
    Occupied(OccupiedEntry<'a, K, V, A>),
}
impl<'a, K, V, A: AllocTuning> Entry<'a, K, V, A>
where
    K: Ord,
{
    /// Get reference to entry key.
    pub fn key(&self) -> &K {
        match self {
            Entry::Vacant(e) => &e.key,
            Entry::Occupied(e) => e.key(),
        }
    }

    /// Insert default value, returning mutable reference to inserted value.
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        match self {
            Entry::Vacant(e) => e.insert(Default::default()),
            Entry::Occupied(e) => e.into_mut(),
        }
    }

    /// Insert value, returning mutable reference to inserted value.
    pub fn or_insert(self, value: V) -> &'a mut V {
        match self {
            Entry::Vacant(e) => e.insert(value),
            Entry::Occupied(e) => e.into_mut(),
        }
    }

    /// Insert default value obtained from function, returning mutable reference to inserted value.
    pub fn or_insert_with<F>(self, default: F) -> &'a mut V
    where
        F: FnOnce() -> V,
    {
        match self {
            Entry::Vacant(e) => e.insert(default()),
            Entry::Occupied(e) => e.into_mut(),
        }
    }

    /// Insert default value obtained from function called with key, returning mutable reference to inserted value.
    pub fn or_insert_with_key<F>(self, default: F) -> &'a mut V
    where
        F: FnOnce(&K) -> V,
    {
        match self {
            Entry::Vacant(e) => {
                let value = default(e.key());
                e.insert(value)
            }
            Entry::Occupied(e) => e.into_mut(),
        }
    }

    /// Modify existing value ( if entry is occupied ).
    pub fn and_modify<F>(mut self, f: F) -> Entry<'a, K, V, A>
    where
        F: FnOnce(&mut V),
    {
        match &mut self {
            Entry::Vacant(_e) => {}
            Entry::Occupied(e) => {
                let v = e.get_mut();
                f(v);
            }
        }
        self
    }
}

/// Vacant [Entry].
pub struct VacantEntry<'a, K, V, A: AllocTuning> {
    key: K,
    cursor: CursorMut<'a, K, V, A>,
}

impl<'a, K: Debug + Ord, V, A: AllocTuning> Debug for VacantEntry<'a, K, V, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("VacantEntry").field(self.key()).finish()
    }
}

impl<'a, K, V, A: AllocTuning> VacantEntry<'a, K, V, A>
where
    K: Ord,
{
    /// Get reference to entry key.
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Get entry key.
    pub fn into_key(self) -> K {
        self.key
    }

    /// Insert value into map returning reference to inserted value.
    pub fn insert(mut self, value: V) -> &'a mut V {
        unsafe { self.cursor.insert_after_unchecked(self.key, value) };
        self.cursor.into_mut()
    }
}

/// Occupied [Entry].
pub struct OccupiedEntry<'a, K, V, A: AllocTuning> {
    cursor: CursorMut<'a, K, V, A>,
}
impl<K: Debug + Ord, V: Debug, A: AllocTuning> Debug for OccupiedEntry<'_, K, V, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OccupiedEntry")
            .field("key", self.key())
            .field("value", self.get())
            .finish()
    }
}

impl<'a, K, V, A: AllocTuning> OccupiedEntry<'a, K, V, A>
where
    K: Ord,
{
    /// Get reference to entry key.
    #[must_use]
    pub fn key(&self) -> &K {
        self.cursor.peek_next().unwrap().0
    }

    /// Remove (key,value) from map, returning key and value.
    #[must_use]
    pub fn remove_entry(mut self) -> (K, V) {
        self.cursor.remove_next().unwrap()
    }

    /// Remove (key,value) from map, returning the value.
    #[must_use]
    pub fn remove(self) -> V {
        self.remove_entry().1
    }

    /// Get reference to the value.
    #[must_use]
    pub fn get(&self) -> &V {
        self.cursor.peek_next().unwrap().1
    }

    /// Get mutable reference to the value.
    pub fn get_mut(&mut self) -> &mut V {
        self.cursor.peek_next().unwrap().1
    }

    /// Get mutable reference to the value, consuming the entry.
    #[must_use]
    pub fn into_mut(self) -> &'a mut V {
        self.cursor.into_mut()
    }

    /// Update the value returns the old value.
    pub fn insert(&mut self, value: V) -> V {
        mem::replace(self.get_mut(), value)
    }
}

// Mutable reference iteration.

enum StealResultMut<'a, K, V> {
    KV((&'a K, &'a mut V)), // Key-value pair.
    CT(&'a mut Tree<K, V>), // Child Tree.
    Nothing,
}

/// Iterator returned by [`BTreeMap::iter_mut`].
#[derive(Debug, Default)]
pub struct IterMut<'a, K, V> {
    len: usize,
    inner: RangeMut<'a, K, V>,
}
impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            self.inner.next()
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}
impl<'a, K, V> ExactSizeIterator for IterMut<'a, K, V> {
    fn len(&self) -> usize {
        self.len
    }
}
impl<'a, K, V> DoubleEndedIterator for IterMut<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            self.inner.next_back()
        }
    }
}
impl<'a, K, V> FusedIterator for IterMut<'a, K, V> {}

#[derive(Debug)]
struct StkMut<'a, K, V> {
    v: IterMutPairVec<'a, K, V>,
    c: std::slice::IterMut<'a, Tree<K, V>>,
}

/// Iterator returned by [`BTreeMap::range_mut`].
#[derive(Debug, Default)]
pub struct RangeMut<'a, K, V> {
    /* There are two iterations going on to implement DoubleEndedIterator.
       fwd_leaf and fwd_stk are initially used for forward (next) iteration,
       once they are exhausted, key-value pairs and child trees are "stolen" from
       bck_stk and bck_leaf which are (conversely) initially used for next_back iteration.
    */
    fwd_leaf: IterMutPairVec<'a, K, V>,
    bck_leaf: IterMutPairVec<'a, K, V>,
    fwd_stk: StkVec<StkMut<'a, K, V>>,
    bck_stk: StkVec<StkMut<'a, K, V>>,
}
impl<'a, K, V> RangeMut<'a, K, V> {
    fn new() -> Self {
        Self {
            fwd_leaf: IterMutPairVec::default(),
            bck_leaf: IterMutPairVec::default(),
            fwd_stk: StkVec::new(),
            bck_stk: StkVec::new(),
        }
    }
    fn push_tree(&mut self, tree: &'a mut Tree<K, V>, both: bool) {
        match tree {
            Tree::L(leaf) => self.fwd_leaf = leaf.iter_mut(),
            Tree::NL(nl) => {
                let (v, mut c) = (nl.v.0.iter_mut(), nl.c.iter_mut());
                let ct = c.next();
                let ct_back = if both { c.next_back() } else { None };
                let both = both && ct_back.is_none();
                self.fwd_stk.push(StkMut { v, c });
                if let Some(ct) = ct {
                    self.push_tree(ct, both);
                }
                if let Some(ct_back) = ct_back {
                    self.push_tree_back(ct_back);
                }
            }
        }
    }
    fn push_range<T, R>(&mut self, tree: &'a mut Tree<K, V>, range: &R, both: bool)
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        match tree {
            Tree::L(leaf) => {
                let (x, y) = leaf.get_xy(range);
                self.fwd_leaf = leaf.0.range_mut(x, y);
            }
            Tree::NL(nl) => {
                let (x, y) = nl.v.get_xy(range);
                let (v, mut c) = (nl.v.0.range_mut(x, y), nl.c[x..=y].iter_mut());

                let ct = c.next();
                let ct_back = if both { c.next_back() } else { None };
                let both = both && ct_back.is_none();

                self.fwd_stk.push(StkMut { v, c });
                if let Some(ct) = ct {
                    self.push_range(ct, range, both);
                }
                if let Some(ct_back) = ct_back {
                    self.push_range_back(ct_back, range);
                }
            }
        }
    }
    fn push_range_back<T, R>(&mut self, tree: &'a mut Tree<K, V>, range: &R)
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        match tree {
            Tree::L(leaf) => {
                let (x, y) = leaf.get_xy(range);
                self.bck_leaf = leaf.0.range_mut(x, y);
            }
            Tree::NL(nl) => {
                let (x, y) = nl.v.get_xy(range);
                let (v, mut c) = (nl.v.0.range_mut(x, y), nl.c[x..=y].iter_mut());
                let ct_back = c.next_back();
                self.bck_stk.push(StkMut { v, c });
                if let Some(ct_back) = ct_back {
                    self.push_range_back(ct_back, range);
                }
            }
        }
    }
    fn push_tree_back(&mut self, tree: &'a mut Tree<K, V>) {
        match tree {
            Tree::L(leaf) => self.bck_leaf = leaf.iter_mut(),
            Tree::NL(nl) => {
                let (v, mut c) = (nl.v.0.iter_mut(), nl.c.iter_mut());
                let ct_back = c.next_back();
                self.bck_stk.push(StkMut { v, c });
                if let Some(ct_back) = ct_back {
                    self.push_tree_back(ct_back);
                }
            }
        }
    }
    fn steal_bck(&mut self) -> StealResultMut<'a, K, V> {
        for s in &mut self.bck_stk {
            if s.v.len() > s.c.len() {
                return StealResultMut::KV(s.v.next().unwrap());
            } else if let Some(ct) = s.c.next() {
                return StealResultMut::CT(ct);
            }
        }
        StealResultMut::Nothing
    }
    fn steal_fwd(&mut self) -> StealResultMut<'a, K, V> {
        for s in &mut self.fwd_stk {
            if s.v.len() > s.c.len() {
                return StealResultMut::KV(s.v.next_back().unwrap());
            } else if let Some(ct) = s.c.next_back() {
                return StealResultMut::CT(ct);
            }
        }
        StealResultMut::Nothing
    }
    #[cold]
    fn next_inner(&mut self) -> Option<(&'a K, &'a mut V)> {
        loop {
            if let Some(s) = self.fwd_stk.last_mut() {
                if let Some(kv) = s.v.next() {
                    if let Some(ct) = s.c.next() {
                        self.push_tree(ct, false);
                    }
                    return Some(kv);
                }
                self.fwd_stk.pop();
            } else {
                match self.steal_bck() {
                    StealResultMut::KV(kv) => return Some(kv),
                    StealResultMut::CT(ct) => {
                        self.push_tree(ct, false);
                        if let Some(x) = self.fwd_leaf.next() {
                            return Some(x);
                        }
                    }
                    StealResultMut::Nothing => {
                        if let Some(x) = self.bck_leaf.next() {
                            return Some(x);
                        }
                        return None;
                    }
                }
            }
        }
    }
    #[cold]
    fn next_back_inner(&mut self) -> Option<(&'a K, &'a mut V)> {
        loop {
            if let Some(s) = self.bck_stk.last_mut() {
                if let Some(kv) = s.v.next_back() {
                    if let Some(ct) = s.c.next_back() {
                        self.push_tree_back(ct);
                    }
                    return Some(kv);
                }
                self.bck_stk.pop();
            } else {
                match self.steal_fwd() {
                    StealResultMut::KV(kv) => return Some(kv),
                    StealResultMut::CT(ct) => {
                        self.push_tree_back(ct);
                        if let Some(x) = self.bck_leaf.next_back() {
                            return Some(x);
                        }
                    }
                    StealResultMut::Nothing => {
                        if let Some(x) = self.fwd_leaf.next_back() {
                            return Some(x);
                        }
                        return None;
                    }
                }
            }
        }
    }
}
impl<'a, K, V> Iterator for RangeMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(x) = self.fwd_leaf.next() {
            return Some(x);
        }
        self.next_inner()
    }
}
impl<'a, K, V> DoubleEndedIterator for RangeMut<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if let Some(x) = self.bck_leaf.next_back() {
            return Some(x);
        }
        self.next_back_inner()
    }
}
impl<'a, K, V> FusedIterator for RangeMut<'a, K, V> {}

// Consuming iteration.

enum StealResultCon<K, V> {
    KV((K, V)),     // Key-value pair.
    CT(Tree<K, V>), // Child Tree.
    Nothing,
}

/// Consuming iterator for [`BTreeMap`].
pub struct IntoIter<K, V> {
    len: usize,
    inner: IntoIterInner<K, V>,
}
impl<K, V> IntoIter<K, V> {
    fn new<A: AllocTuning>(bt: BTreeMap<K, V, A>) -> Self {
        let mut s = Self {
            len: bt.len(),
            inner: IntoIterInner::new(),
        };
        s.inner.push_tree(bt.tree, true);
        s
    }
}
impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        let result = self.inner.next()?;
        self.len -= 1;
        Some(result)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}
impl<K, V> DoubleEndedIterator for IntoIter<K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let result = self.inner.next_back()?;
        self.len -= 1;
        Some(result)
    }
}
impl<K, V> ExactSizeIterator for IntoIter<K, V> {
    fn len(&self) -> usize {
        self.len
    }
}
impl<K, V> FusedIterator for IntoIter<K, V> {}

struct StkCon<K, V> {
    v: IntoIterPairVec<K, V>,
    c: IntoIterShortVec<Tree<K, V>>,
}

struct IntoIterInner<K, V> {
    fwd_leaf: IntoIterPairVec<K, V>,
    bck_leaf: IntoIterPairVec<K, V>,
    fwd_stk: StkVec<StkCon<K, V>>,
    bck_stk: StkVec<StkCon<K, V>>,
}
impl<K, V> IntoIterInner<K, V> {
    fn new() -> Self {
        Self {
            fwd_leaf: IntoIterPairVec::default(),
            bck_leaf: IntoIterPairVec::default(),
            fwd_stk: StkVec::new(),
            bck_stk: StkVec::new(),
        }
    }
    fn push_tree(&mut self, tree: Tree<K, V>, both: bool) {
        match tree {
            Tree::L(leaf) => self.fwd_leaf = leaf.into_iter(),
            Tree::NL(nl) => {
                let (v, mut c) = nl.into_iter();
                let ct = c.next();
                let ct_back = if both { c.next_back() } else { None };
                let both = both && ct_back.is_none();
                self.fwd_stk.push(StkCon { v, c });
                if let Some(ct) = ct {
                    self.push_tree(ct, both);
                }
                if let Some(ct_back) = ct_back {
                    self.push_tree_back(ct_back);
                }
            }
        }
    }
    fn push_tree_back(&mut self, tree: Tree<K, V>) {
        match tree {
            Tree::L(leaf) => self.bck_leaf = leaf.into_iter(),
            Tree::NL(nl) => {
                let (v, mut c) = nl.into_iter();
                let ct_back = c.next_back();
                self.bck_stk.push(StkCon { v, c });
                if let Some(ct_back) = ct_back {
                    self.push_tree_back(ct_back);
                }
            }
        }
    }
    fn steal_bck(&mut self) -> StealResultCon<K, V> {
        for s in &mut self.bck_stk {
            if s.v.len() > s.c.len() {
                return StealResultCon::KV(s.v.next().unwrap());
            } else if let Some(ct) = s.c.next() {
                return StealResultCon::CT(ct);
            }
        }
        StealResultCon::Nothing
    }
    fn steal_fwd(&mut self) -> StealResultCon<K, V> {
        for s in &mut self.fwd_stk {
            if s.v.len() > s.c.len() {
                return StealResultCon::KV(s.v.next_back().unwrap());
            } else if let Some(ct) = s.c.next_back() {
                return StealResultCon::CT(ct);
            }
        }
        StealResultCon::Nothing
    }
    #[cold]
    fn next_inner(&mut self) -> Option<(K, V)> {
        loop {
            if let Some(s) = self.fwd_stk.last_mut() {
                if let Some(kv) = s.v.next() {
                    if let Some(ct) = s.c.next() {
                        self.push_tree(ct, false);
                    }
                    return Some(kv);
                }
                self.fwd_stk.pop();
            } else {
                match self.steal_bck() {
                    StealResultCon::KV(kv) => return Some(kv),
                    StealResultCon::CT(ct) => {
                        self.push_tree(ct, false);
                        if let Some(x) = self.fwd_leaf.next() {
                            return Some(x);
                        }
                    }
                    StealResultCon::Nothing => {
                        if let Some(x) = self.bck_leaf.next() {
                            return Some(x);
                        }
                        return None;
                    }
                }
            }
        }
    }
    #[cold]
    fn next_back_inner(&mut self) -> Option<(K, V)> {
        loop {
            if let Some(s) = self.bck_stk.last_mut() {
                if let Some(kv) = s.v.next_back() {
                    if let Some(ct) = s.c.next_back() {
                        self.push_tree_back(ct);
                    }
                    return Some(kv);
                }
                self.bck_stk.pop();
            } else {
                match self.steal_fwd() {
                    StealResultCon::KV(kv) => return Some(kv),
                    StealResultCon::CT(ct) => {
                        self.push_tree_back(ct);
                        if let Some(x) = self.bck_leaf.next_back() {
                            return Some(x);
                        }
                    }
                    StealResultCon::Nothing => {
                        if let Some(x) = self.fwd_leaf.next_back() {
                            return Some(x);
                        }
                        return None;
                    }
                }
            }
        }
    }
}
impl<K, V> Iterator for IntoIterInner<K, V> {
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(x) = self.fwd_leaf.next() {
            return Some(x);
        }
        self.next_inner()
    }
}
impl<K, V> DoubleEndedIterator for IntoIterInner<K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if let Some(x) = self.bck_leaf.next_back() {
            return Some(x);
        }
        self.next_back_inner()
    }
}

// Immutable reference iteration.

enum StealResult<'a, K, V> {
    KV((&'a K, &'a V)), // Key-value pair.
    CT(&'a Tree<K, V>), // Child Tree.
    Nothing,
}

/// Iterator returned by [`BTreeMap::iter`].
#[derive(Clone, Debug, Default)]
pub struct Iter<'a, K, V> {
    len: usize,
    inner: Range<'a, K, V>,
}
impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            self.inner.next()
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}
impl<'a, K, V> ExactSizeIterator for Iter<'a, K, V> {
    fn len(&self) -> usize {
        self.len
    }
}
impl<'a, K, V> DoubleEndedIterator for Iter<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            self.inner.next_back()
        }
    }
}
impl<'a, K, V> FusedIterator for Iter<'a, K, V> {}

#[derive(Clone, Debug)]
struct Stk<'a, K, V> {
    v: IterPairVec<'a, K, V>,
    c: std::slice::Iter<'a, Tree<K, V>>,
}

/// Iterator returned by [`BTreeMap::range`].
#[derive(Clone, Debug, Default)]
pub struct Range<'a, K, V> {
    fwd_leaf: IterPairVec<'a, K, V>,
    bck_leaf: IterPairVec<'a, K, V>,
    fwd_stk: StkVec<Stk<'a, K, V>>,
    bck_stk: StkVec<Stk<'a, K, V>>,
}
impl<'a, K, V> Range<'a, K, V> {
    fn new() -> Self {
        Self {
            fwd_leaf: IterPairVec::default(),
            bck_leaf: IterPairVec::default(),
            fwd_stk: StkVec::new(),
            bck_stk: StkVec::new(),
        }
    }
    fn push_tree(&mut self, tree: &'a Tree<K, V>, both: bool) {
        match tree {
            Tree::L(leaf) => self.fwd_leaf = leaf.0.iter(),
            Tree::NL(nl) => {
                let (v, mut c) = (nl.v.0.iter(), nl.c.iter());
                let ct = c.next();
                let ct_back = if both { c.next_back() } else { None };
                let both = both && ct_back.is_none();
                self.fwd_stk.push(Stk { v, c });
                if let Some(ct) = ct {
                    self.push_tree(ct, both);
                }
                if let Some(ct_back) = ct_back {
                    self.push_tree_back(ct_back);
                }
            }
        }
    }
    fn push_range<T, R>(&mut self, tree: &'a Tree<K, V>, range: &R, both: bool)
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        match tree {
            Tree::L(leaf) => {
                let (x, y) = leaf.get_xy(range);
                self.fwd_leaf = leaf.0.range(x, y);
            }
            Tree::NL(nl) => {
                let (x, y) = nl.v.get_xy(range);
                let (v, mut c) = (nl.v.0.range(x, y), nl.c[x..=y].iter());

                let ct = c.next();
                let ct_back = if both { c.next_back() } else { None };
                let both = both && ct_back.is_none();

                self.fwd_stk.push(Stk { v, c });
                if let Some(ct) = ct {
                    self.push_range(ct, range, both);
                }
                if let Some(ct_back) = ct_back {
                    self.push_range_back(ct_back, range);
                }
            }
        }
    }
    fn push_range_back<T, R>(&mut self, tree: &'a Tree<K, V>, range: &R)
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        match tree {
            Tree::L(leaf) => {
                let (x, y) = leaf.get_xy(range);
                self.bck_leaf = leaf.0.range(x, y);
            }
            Tree::NL(nl) => {
                let (x, y) = nl.v.get_xy(range);
                let (v, mut c) = (nl.v.0.range(x, y), nl.c[x..=y].iter());
                let ct_back = c.next_back();
                self.bck_stk.push(Stk { v, c });
                if let Some(ct_back) = ct_back {
                    self.push_range_back(ct_back, range);
                }
            }
        }
    }
    fn push_tree_back(&mut self, tree: &'a Tree<K, V>) {
        match tree {
            Tree::L(leaf) => {
                self.bck_leaf = leaf.iter();
            }
            Tree::NL(nl) => {
                let (v, mut c) = (nl.v.0.iter(), nl.c.iter());
                let ct_back = c.next_back();
                self.bck_stk.push(Stk { v, c });
                if let Some(ct_back) = ct_back {
                    self.push_tree_back(ct_back);
                }
            }
        }
    }
    fn steal_bck(&mut self) -> StealResult<'a, K, V> {
        for s in &mut self.bck_stk {
            if s.v.len() > s.c.len() {
                return StealResult::KV(s.v.next().unwrap());
            } else if let Some(ct) = s.c.next() {
                return StealResult::CT(ct);
            }
        }
        StealResult::Nothing
    }
    fn steal_fwd(&mut self) -> StealResult<'a, K, V> {
        for s in &mut self.fwd_stk {
            if s.v.len() > s.c.len() {
                return StealResult::KV(s.v.next_back().unwrap());
            } else if let Some(ct) = s.c.next_back() {
                return StealResult::CT(ct);
            }
        }
        StealResult::Nothing
    }
    #[cold]
    fn next_inner(&mut self) -> Option<(&'a K, &'a V)> {
        loop {
            if let Some(s) = self.fwd_stk.last_mut() {
                if let Some(kv) = s.v.next() {
                    if let Some(ct) = s.c.next() {
                        self.push_tree(ct, false);
                    }
                    return Some(kv);
                }
                self.fwd_stk.pop();
            } else {
                match self.steal_bck() {
                    StealResult::KV(kv) => return Some(kv),
                    StealResult::CT(ct) => {
                        self.push_tree(ct, false);
                        if let Some(x) = self.fwd_leaf.next() {
                            return Some(x);
                        }
                    }
                    StealResult::Nothing => {
                        if let Some(x) = self.bck_leaf.next() {
                            return Some(x);
                        }
                        return None;
                    }
                }
            }
        }
    }
    #[cold]
    fn next_back_inner(&mut self) -> Option<(&'a K, &'a V)> {
        loop {
            if let Some(s) = self.bck_stk.last_mut() {
                if let Some(kv) = s.v.next_back() {
                    if let Some(ct) = s.c.next_back() {
                        self.push_tree_back(ct);
                    }
                    return Some(kv);
                }
                self.bck_stk.pop();
            } else {
                match self.steal_fwd() {
                    StealResult::KV(kv) => return Some(kv),
                    StealResult::CT(ct) => {
                        self.push_tree_back(ct);
                        if let Some(x) = self.bck_leaf.next_back() {
                            return Some(x);
                        }
                    }
                    StealResult::Nothing => {
                        if let Some(x) = self.fwd_leaf.next_back() {
                            return Some(x);
                        }
                        return None;
                    }
                }
            }
        }
    }
}
impl<'a, K, V> Iterator for Range<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(x) = self.fwd_leaf.next() {
            return Some(x);
        }
        self.next_inner()
    }
}
impl<'a, K, V> DoubleEndedIterator for Range<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if let Some(x) = self.bck_leaf.next_back() {
            return Some(x);
        }
        self.next_back_inner()
    }
}
impl<'a, K, V> FusedIterator for Range<'a, K, V> {}

/// Consuming iterator returned by [`BTreeMap::into_keys`].
pub struct IntoKeys<K, V>(IntoIter<K, V>);
impl<K, V> Iterator for IntoKeys<K, V> {
    type Item = K;
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.0.next()?.0)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<K, V> DoubleEndedIterator for IntoKeys<K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(self.0.next_back()?.0)
    }
}
impl<K, V> ExactSizeIterator for IntoKeys<K, V> {
    fn len(&self) -> usize {
        self.0.len()
    }
}
impl<K, V> FusedIterator for IntoKeys<K, V> {}

/// Consuming iterator returned by [`BTreeMap::into_values`].
pub struct IntoValues<K, V>(IntoIter<K, V>);
impl<K, V> Iterator for IntoValues<K, V> {
    type Item = V;
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.0.next()?.1)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<K, V> DoubleEndedIterator for IntoValues<K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(self.0.next_back()?.1)
    }
}
impl<K, V> ExactSizeIterator for IntoValues<K, V> {
    fn len(&self) -> usize {
        self.0.len()
    }
}
impl<K, V> FusedIterator for IntoValues<K, V> {}

// Trivial iterators.

/// Iterator returned by [`BTreeMap::values_mut`].
#[derive(Debug)]
pub struct ValuesMut<'a, K, V>(IterMut<'a, K, V>);
impl<'a, K, V> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(_, v)| v)
    }
}
impl<'a, K, V> DoubleEndedIterator for ValuesMut<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|(_, v)| v)
    }
}
impl<'a, K, V> ExactSizeIterator for ValuesMut<'a, K, V> {
    fn len(&self) -> usize {
        self.0.len()
    }
}
impl<'a, K, V> FusedIterator for ValuesMut<'a, K, V> {}

/// Iterator returned by [`BTreeMap::values`].
#[derive(Clone, Debug, Default)]
pub struct Values<'a, K, V>(Iter<'a, K, V>);
impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(_, v)| v)
    }
}
impl<'a, K, V> DoubleEndedIterator for Values<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|(_, v)| v)
    }
}
impl<'a, K, V> ExactSizeIterator for Values<'a, K, V> {
    fn len(&self) -> usize {
        self.0.len()
    }
}
impl<'a, K, V> FusedIterator for Values<'a, K, V> {}

/// Iterator returned by [`BTreeMap::keys`].
#[derive(Clone, Debug, Default)]
pub struct Keys<'a, K, V>(Iter<'a, K, V>);
impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(k, _)| k)
    }
}
impl<'a, K, V> DoubleEndedIterator for Keys<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|(k, _)| k)
    }
}
impl<'a, K, V> ExactSizeIterator for Keys<'a, K, V> {
    fn len(&self) -> usize {
        self.0.len()
    }
}
impl<'a, K, V> FusedIterator for Keys<'a, K, V> {}

/// Iterator returned by [`BTreeMap::extract_if`].
// #[derive(Debug)]
pub struct ExtractIf<'a, K, V, A: AllocTuning, F>
where
    F: FnMut(&K, &mut V) -> bool,
{
    source: CursorMut<'a, K, V, A>,
    pred: F,
}
impl<K, V, A: AllocTuning, F> fmt::Debug for ExtractIf<'_, K, V, A, F>
where
    K: fmt::Debug,
    V: fmt::Debug,
    F: FnMut(&K, &mut V) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ExtractIf")
            .field(&self.source.peek_next())
            .finish()
    }
}
impl<'a, K, V, A: AllocTuning, F> Iterator for ExtractIf<'a, K, V, A, F>
where
    F: FnMut(&K, &mut V) -> bool,
{
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (k, v) = self.source.peek_next()?;
            if (self.pred)(k, v) {
                return self.source.remove_next();
            }
            self.source.next();
        }
    }
}
impl<'a, K, V, A: AllocTuning, F> FusedIterator for ExtractIf<'a, K, V, A, F> where
    F: FnMut(&K, &mut V) -> bool
{
}

// Cursors.

/// Error type for [`CursorMut::insert_before`] and [`CursorMut::insert_after`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct UnorderedKeyError {}
impl fmt::Display for UnorderedKeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "key is not properly ordered relative to neighbors")
    }
}
impl std::error::Error for UnorderedKeyError {}

/// Cursor that allows mutation of map, returned by [`BTreeMap::lower_bound_mut`], [`BTreeMap::upper_bound_mut`].
pub struct CursorMut<'a, K, V, A: AllocTuning>(CursorMutKey<'a, K, V, A>);
impl<'a, K, V, A: AllocTuning> CursorMut<'a, K, V, A> {
    fn lower_bound<Q>(map: &'a mut BTreeMap<K, V, A>, bound: Bound<&Q>) -> Self
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        unsafe {
            // Converting map to raw pointer here is necessary to keep Miri happy
            // although not when using MIRIFLAGS=-Zmiri-tree-borrows.
            let map: *mut BTreeMap<K, V, A> = map;
            let mut s = CursorMutKey::make(map);
            s.push_lower(&mut (*map).tree, bound);
            Self(s)
        }
    }

    fn upper_bound<Q>(map: &'a mut BTreeMap<K, V, A>, bound: Bound<&Q>) -> Self
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        unsafe {
            let map: *mut BTreeMap<K, V, A> = map;
            let mut s = CursorMutKey::make(map);
            s.push_upper(&mut (*map).tree, bound);
            Self(s)
        }
    }

    /// Insert leaving cursor after newly inserted element.
    pub fn insert_before(&mut self, key: K, value: V) -> Result<(), UnorderedKeyError>
    where
        K: Ord,
    {
        self.0.insert_before(key, value)
    }

    /// Insert leaving cursor before newly inserted element.
    pub fn insert_after(&mut self, key: K, value: V) -> Result<(), UnorderedKeyError>
    where
        K: Ord,
    {
        self.0.insert_after(key, value)
    }

    /// Insert leaving cursor after newly inserted element.
    /// # Safety
    ///
    /// Keys must be unique and in sorted order.
    pub unsafe fn insert_before_unchecked(&mut self, key: K, value: V) {
        self.0.insert_before_unchecked(key, value);
    }

    /// Insert leaving cursor before newly inserted element.
    /// # Safety
    ///
    /// Keys must be unique and in sorted order.
    pub unsafe fn insert_after_unchecked(&mut self, key: K, value: V) {
        self.0.insert_after_unchecked(key, value);
    }

    /// Remove previous element.
    pub fn remove_prev(&mut self) -> Option<(K, V)> {
        self.0.remove_prev()
    }

    /// Remove next element.
    pub fn remove_next(&mut self) -> Option<(K, V)> {
        self.0.remove_next()
    }

    /// Advance the cursor, returns references to the key and value of the element that it moved over.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<(&K, &mut V)> {
        let (k, v) = self.0.next()?;
        Some((&*k, v))
    }

    /// Move the cursor back, returns references to the key and value of the element that it moved over.
    pub fn prev(&mut self) -> Option<(&K, &mut V)> {
        let (k, v) = self.0.prev()?;
        Some((&*k, v))
    }

    /// Get references to the next key/value pair.
    #[must_use]
    pub fn peek_next(&self) -> Option<(&K, &mut V)> {
        let (k, v) = self.0.peek_next()?;
        Some((&*k, v))
    }

    /// Get references to the previous key/value pair.
    #[must_use]
    pub fn peek_prev(&self) -> Option<(&K, &mut V)> {
        let (k, v) = self.0.peek_prev()?;
        Some((&*k, v))
    }

    /// Converts the cursor into a `CursorMutKey`, which allows mutating the key of elements in the tree.
    /// # Safety
    ///
    /// Keys must be unique and in sorted order.
    #[must_use]
    pub unsafe fn with_mutable_key(self) -> CursorMutKey<'a, K, V, A> {
        self.0
    }

    /// Returns a read-only cursor pointing to the same location as the `CursorMut`.
    #[must_use]
    pub fn as_cursor(&self) -> Cursor<'_, K, V, A> {
        self.0.as_cursor()
    }

    /// This is needed for the implementation of the [Entry] API.
    fn into_mut(self) -> &'a mut V {
        self.0.into_mut()
    }
}

/// Cursor that allows mutation of map keys, returned by [`CursorMut::with_mutable_key`].
pub struct CursorMutKey<'a, K, V, A: AllocTuning> {
    map: *mut BTreeMap<K, V, A>,
    leaf: Option<*mut Leaf<K, V>>,
    index: usize,
    stack: StkVec<(*mut NonLeaf<K, V>, usize)>,
    _pd: PhantomData<&'a mut BTreeMap<K, V, A>>,
}

unsafe impl<'a, K, V, A: AllocTuning> Send for CursorMutKey<'a, K, V, A> {}
unsafe impl<'a, K, V, A: AllocTuning> Sync for CursorMutKey<'a, K, V, A> {}

impl<'a, K, V, A: AllocTuning> CursorMutKey<'a, K, V, A> {
    fn make(map: *mut BTreeMap<K, V, A>) -> Self {
        Self {
            map,
            leaf: None,
            index: 0,
            stack: StkVec::new(),
            _pd: PhantomData,
        }
    }

    fn push_lower<Q>(&mut self, mut tree: &mut Tree<K, V>, bound: Bound<&Q>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        loop {
            match tree {
                Tree::L(leaf) => {
                    self.index = leaf.get_lower(bound);
                    self.leaf = Some(leaf);
                    break;
                }
                Tree::NL(nl) => {
                    let ix = nl.v.get_lower(bound);
                    self.stack.push((nl, ix));
                    tree = nl.c.ixm(ix);
                }
            }
        }
    }

    fn push_upper<Q>(&mut self, mut tree: &mut Tree<K, V>, bound: Bound<&Q>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        loop {
            match tree {
                Tree::L(leaf) => {
                    self.index = leaf.get_upper(bound);
                    self.leaf = Some(leaf);
                    break;
                }
                Tree::NL(nl) => {
                    let ix = nl.v.get_upper(bound);
                    self.stack.push((nl, ix));
                    tree = nl.c.ixm(ix);
                }
            }
        }
    }

    fn push(&mut self, mut tsp: usize, mut tree: &mut Tree<K, V>) {
        loop {
            match tree {
                Tree::L(leaf) => {
                    self.index = 0;
                    self.leaf = Some(leaf);
                    break;
                }
                Tree::NL(nl) => {
                    self.stack[tsp] = (nl, 0);
                    tree = nl.c.ixm(0);
                    tsp += 1;
                }
            }
        }
    }

    fn push_back(&mut self, mut tsp: usize, mut tree: &mut Tree<K, V>) {
        loop {
            match tree {
                Tree::L(leaf) => {
                    self.index = leaf.0.len();
                    self.leaf = Some(leaf);
                    break;
                }
                Tree::NL(nl) => {
                    let ix = nl.v.0.len();
                    self.stack[tsp] = (nl, ix);
                    tree = nl.c.ixm(ix);
                    tsp += 1;
                }
            }
        }
    }

    /// Insert leaving cursor after newly inserted element.
    pub fn insert_before(&mut self, key: K, value: V) -> Result<(), UnorderedKeyError>
    where
        K: Ord,
    {
        if let Some((prev, _)) = self.peek_prev() {
            if &key <= prev {
                return Err(UnorderedKeyError {});
            }
        }
        if let Some((next, _)) = self.peek_next() {
            if &key >= next {
                return Err(UnorderedKeyError {});
            }
        }
        unsafe {
            self.insert_before_unchecked(key, value);
        }
        Ok(())
    }

    /// Insert leaving cursor before newly inserted element.
    pub fn insert_after(&mut self, key: K, value: V) -> Result<(), UnorderedKeyError>
    where
        K: Ord,
    {
        if let Some((prev, _)) = self.peek_prev() {
            if &key <= prev {
                return Err(UnorderedKeyError {});
            }
        }
        if let Some((next, _)) = self.peek_next() {
            if &key >= next {
                return Err(UnorderedKeyError {});
            }
        }
        unsafe {
            self.insert_after_unchecked(key, value);
        }
        Ok(())
    }

    /// Insert leaving cursor after newly inserted element.
    /// # Safety
    ///
    /// Keys must be unique and in sorted order.
    pub unsafe fn insert_before_unchecked(&mut self, key: K, value: V) {
        self.insert_after_unchecked(key, value);
        self.index += 1;
    }

    /// Insert leaving cursor before newly inserted element.
    /// # Safety
    ///
    /// Keys must be unique and in sorted order.
    pub unsafe fn insert_after_unchecked(&mut self, key: K, value: V) {
        unsafe {
            (*self.map).len += 1;
            let mut leaf = self.leaf.unwrap_unchecked();
            if (*leaf).full() {
                let a = &(*self.map).atune;
                match a.full_action(self.index, (*leaf).0.len()) {
                    FullAction::Split(b, a1, a2) => {
                        let (med, right) = (*leaf).0.split(b, a1, a2);
                        let right = Tree::L(Leaf(right));
                        let r = usize::from(self.index > b);
                        self.index -= r * (b + 1);
                        assert!(self.index < if r == 1 { a2 } else { a1 });
                        let t = self.save_split(med, right, r);
                        leaf = (*t).leaf();
                        self.leaf = Some(leaf);
                    }
                    FullAction::Extend(to) => (*leaf).0.set_alloc(to),
                }
            }
            (*leaf).0.insert(self.index, (key, value));
        }
    }

    fn save_split(&mut self, med: (K, V), tree: Tree<K, V>, r: usize) -> *mut Tree<K, V> {
        unsafe {
            if let Some((mut nl, mut ix)) = self.stack.pop() {
                if (*nl).v.full() {
                    let a = &(*self.map).atune;
                    match a.full_action(ix, (*nl).v.0.len()) {
                        FullAction::Split(b, a1, a2) => {
                            let (med, right) = (*nl).split(b, a1, a2);
                            let r = usize::from(ix > b);
                            ix -= r * (b + 1);
                            assert!(ix < if r == 1 { a2 } else { a1 });
                            let t = self.save_split(med, Tree::NL(right), r);
                            nl = (*t).nonleaf();
                        }
                        FullAction::Extend(to) => {
                            (*nl).v.0.set_alloc(to);
                            (*nl).c.set_alloc(to + 1);
                        }
                    }
                }
                (*nl).v.0.insert(ix, med);
                (*nl).c.insert(ix + 1, tree);
                ix += r;
                self.stack.push((nl, ix));
                (*nl).c.ixm(ix)
            } else {
                (*self.map).tree.new_root((med, tree));
                let nl = (*self.map).tree.nonleaf();
                self.stack.push((nl, r));
                nl.c.ixm(r)
            }
        }
    }

    /// Remove previous element.
    pub fn remove_prev(&mut self) -> Option<(K, V)> {
        self.prev()?;
        self.remove_next()
    }

    /// Remove next element.
    pub fn remove_next(&mut self) -> Option<(K, V)> {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            if self.index == (*leaf).0.len() {
                let mut tsp = self.stack.len();
                while tsp > 0 {
                    tsp -= 1;
                    let (nl, ix) = self.stack[tsp];
                    if ix < (*nl).v.0.len() {
                        let a = &(*self.map).atune;
                        let (kv, ix) = (*nl).remove_at(ix, a);
                        self.stack[tsp] = (nl, ix);
                        self.push(tsp + 1, (*nl).c.ixm(ix));
                        (*self.map).len -= 1;
                        return Some(kv);
                    }
                }
                None
            } else {
                (*self.map).len -= 1;
                Some((*leaf).0.remove(self.index))
            }
        }
    }

    /// Advance the cursor, returns references to the key and value of the element that it moved over.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<(&mut K, &mut V)> {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            if self.index == (*leaf).0.len() {
                let mut tsp = self.stack.len();
                while tsp > 0 {
                    tsp -= 1;
                    let (nl, mut ix) = self.stack[tsp];
                    if ix < (*nl).v.0.len() {
                        let kv = (*nl).v.0.ixbm(ix);
                        ix += 1;
                        self.stack[tsp] = (nl, ix);
                        self.push(tsp + 1, (*nl).c.ixm(ix));
                        return Some(kv);
                    }
                }
                None
            } else {
                let kv = (*leaf).0.ixbm(self.index);
                self.index += 1;
                Some(kv)
            }
        }
    }

    /// Move the cursor back, returns references to the key and value of the element that it moved over.
    pub fn prev(&mut self) -> Option<(&mut K, &mut V)> {
        unsafe {
            if self.index == 0 {
                let mut tsp = self.stack.len();
                while tsp > 0 {
                    tsp -= 1;
                    let (nl, mut ix) = self.stack[tsp];
                    if ix > 0 {
                        ix -= 1;
                        let kv = (*nl).v.0.ixbm(ix);
                        self.stack[tsp] = (nl, ix);
                        self.push_back(tsp + 1, (*nl).c.ixm(ix));
                        return Some(kv);
                    }
                }
                None
            } else {
                let leaf = self.leaf.unwrap_unchecked();
                self.index -= 1;
                Some((*leaf).0.ixbm(self.index))
            }
        }
    }

    /// Returns mutable references to the next key/value pair.
    #[must_use]
    pub fn peek_next(&self) -> Option<(&mut K, &mut V)> {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            if self.index == (*leaf).0.len() {
                for (nl, ix) in self.stack.iter().rev() {
                    if *ix < (**nl).v.0.len() {
                        return Some((**nl).v.0.ixbm(*ix));
                    }
                }
                None
            } else {
                Some((*leaf).0.ixbm(self.index))
            }
        }
    }
    /// Returns mutable references to the previous key/value pair.
    #[must_use]
    pub fn peek_prev(&self) -> Option<(&mut K, &mut V)> {
        unsafe {
            if self.index == 0 {
                for (nl, ix) in self.stack.iter().rev() {
                    if *ix > 0 {
                        return Some((**nl).v.0.ixbm(*ix - 1));
                    }
                }
                None
            } else {
                let leaf = self.leaf.unwrap_unchecked();
                Some((*leaf).0.ixbm(self.index - 1))
            }
        }
    }

    /// Returns a read-only cursor pointing to the same location as the `CursorMutKey`.
    #[must_use]
    pub fn as_cursor(&self) -> Cursor<'_, K, V, A> {
        unsafe {
            let mut c = Cursor::make();
            c.index = self.index;
            c.leaf = Some(&*self.leaf.unwrap());
            for (nl, ix) in &self.stack {
                c.stack.push((&(**nl), *ix));
            }
            c
        }
    }

    /// This is needed for the implementation of the [Entry] API.
    fn into_mut(self) -> &'a mut V {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            (*leaf).0.ixmv(self.index)
        }
    }
}

/// Cursor returned by [`BTreeMap::lower_bound`], [`BTreeMap::upper_bound`].
#[derive(Debug, Clone)]
pub struct Cursor<'a, K, V, A: AllocTuning> {
    leaf: Option<*const Leaf<K, V>>,
    index: usize,
    stack: StkVec<(*const NonLeaf<K, V>, usize)>,
    _pd: PhantomData<&'a BTreeMap<K, V, A>>,
}

unsafe impl<'a, K, V, A: AllocTuning> Send for Cursor<'a, K, V, A> {}
unsafe impl<'a, K, V, A: AllocTuning> Sync for Cursor<'a, K, V, A> {}

impl<'a, K, V, A: AllocTuning> Cursor<'a, K, V, A> {
    fn make() -> Self {
        Self {
            leaf: None,
            index: 0,
            stack: StkVec::new(),
            _pd: PhantomData,
        }
    }

    fn lower_bound<Q>(bt: &'a BTreeMap<K, V, A>, bound: Bound<&Q>) -> Self
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let mut s = Self::make();
        s.push_lower(&bt.tree, bound);
        s
    }

    fn upper_bound<Q>(bt: &'a BTreeMap<K, V, A>, bound: Bound<&Q>) -> Self
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let mut s = Self::make();
        s.push_upper(&bt.tree, bound);
        s
    }

    fn push_lower<Q>(&mut self, mut tree: &'a Tree<K, V>, bound: Bound<&Q>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        loop {
            match tree {
                Tree::L(leaf) => {
                    self.leaf = Some(leaf);
                    self.index = leaf.get_lower(bound);
                    break;
                }
                Tree::NL(nl) => {
                    let ix = nl.v.get_lower(bound);
                    self.stack.push((nl, ix));
                    tree = nl.c.ix(ix);
                }
            }
        }
    }

    fn push_upper<Q>(&mut self, mut tree: &'a Tree<K, V>, bound: Bound<&Q>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        loop {
            match tree {
                Tree::L(leaf) => {
                    self.leaf = Some(leaf);
                    self.index = leaf.get_upper(bound);
                    break;
                }
                Tree::NL(nl) => {
                    let ix = nl.v.get_upper(bound);
                    self.stack.push((nl, ix));
                    tree = nl.c.ix(ix);
                }
            }
        }
    }

    fn push(&mut self, mut tsp: usize, mut tree: &Tree<K, V>) {
        loop {
            match tree {
                Tree::L(leaf) => {
                    self.leaf = Some(leaf);
                    self.index = 0;
                    break;
                }
                Tree::NL(nl) => {
                    self.stack[tsp] = (nl, 0);
                    tree = nl.c.ix(0);
                    tsp += 1;
                }
            }
        }
    }

    fn push_back(&mut self, mut tsp: usize, mut tree: &Tree<K, V>) {
        loop {
            match tree {
                Tree::L(leaf) => {
                    self.leaf = Some(leaf);
                    self.index = leaf.0.len();
                    break;
                }
                Tree::NL(nl) => {
                    let ix = nl.v.0.len();
                    self.stack[tsp] = (nl, ix);
                    tree = nl.c.ix(ix);
                    tsp += 1;
                }
            }
        }
    }

    /// Advance the cursor, returns references to the key and value of the element that it moved over.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<(&K, &V)> {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            if self.index == (*leaf).0.len() {
                let mut tsp = self.stack.len();
                while tsp > 0 {
                    tsp -= 1;
                    let (nl, mut ix) = self.stack[tsp];
                    if ix < (*nl).v.0.len() {
                        let kv = (*nl).v.0.ix(ix);
                        ix += 1;
                        self.stack[tsp] = (nl, ix);
                        let ct = (*nl).c.ix(ix);
                        self.push(tsp + 1, ct);
                        return Some(kv);
                    }
                }
                None
            } else {
                let kv = (*leaf).0.ix(self.index);
                self.index += 1;
                Some(kv)
            }
        }
    }

    /// Move the cursor back, returns references to the key and value of the element that it moved over.
    pub fn prev(&mut self) -> Option<(&K, &V)> {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            if self.index == 0 {
                let mut tsp = self.stack.len();
                while tsp > 0 {
                    tsp -= 1;
                    let (nl, mut ix) = self.stack[tsp];
                    if ix > 0 {
                        ix -= 1;
                        let kv = (*nl).v.0.ix(ix);
                        self.stack[tsp] = (nl, ix);
                        let ct = (*nl).c.ix(ix);
                        self.push_back(tsp + 1, ct);
                        return Some(kv);
                    }
                }
                None
            } else {
                self.index -= 1;
                Some((*leaf).0.ix(self.index))
            }
        }
    }

    /// Returns references to the next key/value pair.
    #[must_use]
    pub fn peek_next(&self) -> Option<(&K, &V)> {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            if self.index == (*leaf).0.len() {
                for (nl, ix) in self.stack.iter().rev() {
                    if *ix < (**nl).v.0.len() {
                        return Some((**nl).v.0.ix(*ix));
                    }
                }
                None
            } else {
                Some((*leaf).0.ix(self.index))
            }
        }
    }
    /// Returns references to the previous key/value pair.
    #[must_use]
    pub fn peek_prev(&self) -> Option<(&K, &V)> {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            if self.index == 0 {
                for (nl, ix) in self.stack.iter().rev() {
                    if *ix > 0 {
                        return Some((**nl).v.0.ix(*ix - 1));
                    }
                }
                None
            } else {
                Some((*leaf).0.ix(self.index - 1))
            }
        }
    }
}

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
#[cfg(all(test, not(miri), not(feature = "cap")))]
use mimalloc::MiMalloc;

#[cfg(all(test, not(miri), not(feature = "cap")))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[cfg(test)]
mod mytests;

// #[cfg(test)]
// mod stdtests; // Increases compile/link time to 9 seconds from 3 seconds, so sometimes commented out!
