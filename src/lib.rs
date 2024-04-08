//! This crate implements a BTreeMap similar to std::collections::BTreeMap.
//!
//! One difference is that keys (as well as values) can be mutated using mutable iterators.
//!
//! Note: some (crate) private methods of FixedCapVec are techically unsafe in release mode when the unsafe_optim feature is enabled, but are not declared as such to avoid littering the code with unsafe blocks.

#![deny(missing_docs)]
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::iter::FusedIterator;
use std::ops::{Bound, RangeBounds};

// Vector types.
mod vecs;
use vecs::FixedCapVec;
type LeafVec<K, V> = FixedCapVec<LEAF_FULL, (K, V)>;
type NonLeafVec<K, V> = FixedCapVec<NON_LEAF_FULL, (K, V)>;
type NonLeafChildVec<K, V> = FixedCapVec<{ NON_LEAF_FULL + 1 }, Tree<K, V>>;
type PosVec = smallvec::SmallVec<[u8; 8]>;

type Split<K, V> = ((K, V), Tree<K, V>);

const LEAF_SPLIT: usize = 20;
const LEAF_FULL: usize = LEAF_SPLIT * 2 - 1;
const NON_LEAF_SPLIT: usize = 30;
const NON_LEAF_FULL: usize = NON_LEAF_SPLIT * 2 - 1;

fn bounded<T, R>(range: &R) -> (bool, bool)
where
    T: Ord + ?Sized,
    R: RangeBounds<T>,
{
    use Bound::*;
    let (left, right) = match (range.start_bound(), range.end_bound()) {
        (Unbounded, Unbounded) => (false, false),
        (Unbounded, Included(_)) => (false, true),
        (Unbounded, Excluded(_)) => (false, true),
        (Included(_), Unbounded) => (true, false),
        (Included(s), Included(e)) => {
            if e < s {
                panic!("range start is greater than range end in BTreeMap")
            }
            (true, true)
        }
        (Included(s), Excluded(e)) => {
            if e < s {
                panic!("range start is greater than range end in BTreeMap")
            }
            (true, true)
        }
        (Excluded(_), Unbounded) => (true, false),
        (Excluded(s), Included(e)) => {
            if e < s {
                panic!("range start is greater than range end in BTreeMap")
            }
            (true, true)
        }
        (Excluded(s), Excluded(e)) => {
            if e == s {
                panic!("range start and end are equal and excluded in BTreeMap")
            }
            if e < s {
                panic!("range start is greater than range end in BTreeMap")
            }
            (true, true)
        }
    };
    (left, right)
}

/// BTreeMap similar to [std::collections::BTreeMap].
pub struct BTreeMap<K, V> {
    len: usize,
    tree: Tree<K, V>,
}

impl<K, V> Default for BTreeMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> BTreeMap<K, V> {
    #[cfg(test)]
    fn check(&self) {}

    /// Returns a new, empty map.
    pub fn new() -> Self {
        Self {
            len: 0,
            tree: Tree::default(),
        }
    }

    /// Clear the map.
    pub fn clear(&mut self) {
        self.len = 0;
        self.tree = Tree::default();
    }

    /// Get number of key-value pairs in the map.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Is the map empty?
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get Entry for map key.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V>
    where
        K: Ord,
    {
        let mut pos = Position::new();
        self.tree.find_position(&key, &mut pos);
        if pos.key_found {
            let key = OccupiedEntryKey::Some(pos);
            Entry::Occupied(OccupiedEntry { map: self, key })
        } else {
            Entry::Vacant(VacantEntry {
                map: self,
                key,
                pos,
            })
        }
    }

    /// Get first Entry.
    pub fn first_entry(&mut self) -> Option<OccupiedEntry<'_, K, V>> {
        if self.is_empty() {
            None
        } else {
            Some(OccupiedEntry {
                map: self,
                key: OccupiedEntryKey::First,
            })
        }
    }

    /// Get last Entry.
    pub fn last_entry(&mut self) -> Option<OccupiedEntry<'_, K, V>> {
        if self.is_empty() {
            None
        } else {
            Some(OccupiedEntry {
                map: self,
                key: OccupiedEntryKey::Last,
            })
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

    /// Does the map have an entry for the specified key.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.get_key_value(key).is_some()
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
        let result = self.tree.remove(key);
        if result.is_some() {
            self.len -= 1;
        }
        result
    }

    /// Remove first key-value pair from map.
    pub fn pop_first(&mut self) -> Option<(K, V)> {
        let result = self.tree.pop_first();
        if result.is_some() {
            self.len -= 1;
        }
        result
    }

    /// Remove last key-value pair from map.
    pub fn pop_last(&mut self) -> Option<(K, V)> {
        let result = self.tree.pop_last();
        if result.is_some() {
            self.len -= 1;
        }
        result
    }

    /// Remove all key-value pairs, visited in ascending order, for which f returns false.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        self.len -= self.tree.retain(&mut f);
    }

    /// Get reference to the value corresponding to the key.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.get_key_value(key).map(|(_k, v)| v)
    }

    /// Get a mutable reference to the value corresponding to the key.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        if let Some((_k, v)) = self.tree.get_mut(key) {
            Some(v)
        } else {
            None
        }
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
    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        self.tree.iter().next()
    }

    /// Gets references to last key and value.
    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        self.tree.iter().next_back()
    }

    /// Get mutable references to first key and value.
    pub fn first_key_value_mut(&mut self) -> Option<(&mut K, &mut V)> {
        self.tree.iter_mut().next()
    }

    /// Get mutable references to last key and value.
    pub fn last_key_value_mut(&mut self) -> Option<(&mut K, &mut V)> {
        self.tree.iter_mut().next_back()
    }

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    ///
    /// If a key from `other` is already present in `self`, the respective
    /// value from `self` will be overwritten with the respective value from `other`.
    pub fn append(&mut self, other: &mut BTreeMap<K, V>)
    where
        K: Ord,
    {
        while let Some((k, v)) = other.pop_first() {
            self.insert(k, v);
        }
        other.clear();
    }

    /// Splits the collection into two at the given key.
    /// Returns everything after the given key, including the key.
    pub fn split_off<Q: ?Sized + Ord>(&mut self, key: &Q) -> Self
    where
        K: Borrow<Q> + Ord,
    {
        let mut map = Self::new();
        while let Some((k, v)) = self.pop_last() {
            if k.borrow() < key {
                self.insert(k, v);
                break;
            }
            map.insert(k, v);
        }
        map
    }

    /// Get iterator of references to key-value pairs.
    pub fn iter(&self) -> Iter<'_, K, V> {
        self.tree.iter()
    }

    /// Get iterator of mutable references to key-value pairs.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        self.tree.iter_mut()
    }

    /// Get iterator for range of references to key-value pairs.
    pub fn range<T, R>(&self, range: R) -> Iter<'_, K, V>
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        let (left, right) = bounded(&range);
        self.tree.range(&range, left, right)
    }

    /// Get iterator for range of mutable references to key-value pairs.
    /// A key can be mutated, provided it does not change the map order.
    pub fn range_mut<T, R>(&mut self, range: R) -> IterMut<'_, K, V>
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        let (left, right) = bounded(&range);
        self.tree.range_mut(&range, left, right)
    }

    /// Get iterator of references to keys.
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys(self.iter())
    }

    /// Get iterator of references to values.
    pub fn values(&self) -> Values<'_, K, V> {
        Values(self.iter())
    }

    /// Get iterator of mutable references to values.
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        ValuesMut(self.iter_mut())
    }

    /// Get consuming iterator that returns all the keys, in sorted order.
    pub fn into_keys(self) -> IntoKeys<K, V> {
        IntoKeys(self)
    }

    /// Get consuming iterator that returns all the values, in sorted order.
    pub fn into_values(self) -> IntoValues<K, V> {
        IntoValues(self)
    }

    /// Walk the map in sorted order, calling action with reference to key-value pair for each key >= start.
    /// If action returns true the walk terminates.
    pub fn walk<F, Q>(&self, start: &Q, action: &mut F) -> bool
    where
        F: FnMut(&(K, V)) -> bool,
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.tree.walk(start, action)
    }

    /// Walk the map in sorted order, calling action with mutable reference to key-value pair for each key >= start.
    /// If action returns true the walk terminates.
    /// The key can be mutated by action if it does not change the map order.
    pub fn walk_mut<F, Q>(&mut self, start: &Q, action: &mut F) -> bool
    where
        F: FnMut(&mut (K, V)) -> bool,
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.tree.walk_mut(start, action)
    }

    fn ins_pos(&mut self, pos: &mut Position<K, V>, key: K, value: V) -> &mut (K, V) {
        if !pos.leaf_has_space {
            if let Some(s) = self.tree.prepare_insert(&mut pos.ix, 0) {
                self.tree.new_root(s);
            }
        }
        self.len += 1;
        self.tree.do_insert(&pos.ix, 0, key, value)
    }
} // End impl BTreeMap

use std::hash::{Hash, Hasher};
impl<K: Hash, V: Hash> Hash for BTreeMap<K, V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // state.write_length_prefix(self.len());
        for elt in self.iter() {
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

impl<K, V> IntoIterator for BTreeMap<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    /// Convert BTreeMap to Iterator.
    fn into_iter(self) -> IntoIter<K, V> {
        IntoIter(self)
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
    type Item = (&'a mut K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;
    fn into_iter(self) -> IterMut<'a, K, V> {
        self.iter_mut()
    }
}

impl<K, V> Clone for BTreeMap<K, V>
where
    K: Clone + Ord,
    V: Clone,
{
    fn clone(&self) -> BTreeMap<K, V> {
        let mut map = BTreeMap::new();
        for (k, v) in self.iter() {
            map.insert(k.clone(), v.clone());
        }
        map
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
    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

use std::fmt;
use std::fmt::Debug;
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
use std::marker::PhantomData;

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
    // The type that our Visitor is going to produce.
    type Value = BTreeMap<K, V>;

    // Format a message stating what data this Visitor expects to receive.
    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("BTreeMap")
    }

    // Deserialize MyMap from an abstract "map" provided by the
    // Deserializer. The MapAccess input is a callback provided by
    // the Deserializer to let us see each entry in the map.
    fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut map = BTreeMap::new();

        while let Some((key, value)) = access.next_entry()? {
            map.insert(key, value);
        }

        Ok(map)
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
        // Instantiate our Visitor and ask the Deserializer to drive
        // it over the input data, resulting in an instance of MyMap.
        deserializer.deserialize_map(BTreeMapVisitor::new())
    }
}

struct InsertCtx<K, V> {
    value: Option<V>,
    split: Option<Split<K, V>>,
}

/// Entry in BTreeMap, returned by [BTreeMap::entry].
pub enum Entry<'a, K, V> {
    /// Vacant entry - map doesn't yet contain key.
    Vacant(VacantEntry<'a, K, V>),
    /// Occupied entry - map already contains key.
    Occupied(OccupiedEntry<'a, K, V>),
}

impl<'a, K, V> Entry<'a, K, V>
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
    pub fn and_modify<F>(mut self, f: F) -> Entry<'a, K, V>
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

enum TreePtr<K, V> {
    None,
    L(*mut Leaf<K, V>, usize),
    NL(*mut NonLeaf<K, V>, usize),
}

unsafe impl<K: Send, V: Send> Send for TreePtr<K, V> {}
unsafe impl<K: Sync, V: Send> Sync for TreePtr<K, V> {}

impl<K, V> TreePtr<K, V> {
    fn value_mut(&mut self) -> &mut V {
        match self {
            TreePtr::None => panic!(),
            TreePtr::L(ptr, ix) => unsafe { &mut (*(*ptr)).0.ixm(*ix).1 },
            TreePtr::NL(ptr, ix) => unsafe { &mut (*(*ptr)).v.ixm(*ix).1 },
        }
    }

    fn value_ref(&self) -> &V {
        match self {
            TreePtr::None => panic!(),
            TreePtr::L(ptr, ix) => unsafe { &(*(*ptr)).0.ix(*ix).1 },
            TreePtr::NL(ptr, ix) => unsafe { &(*(*ptr)).v.ix(*ix).1 },
        }
    }

    fn key_ref(&self) -> &K {
        match self {
            TreePtr::None => panic!(),
            TreePtr::L(ptr, ix) => unsafe { &(*(*ptr)).0.ix(*ix).0 },
            TreePtr::NL(ptr, ix) => unsafe { &(*(*ptr)).v.ix(*ix).0 },
        }
    }
}

/// Represents position of key in Btree.
struct Position<K, V> {
    key_found: bool,
    leaf_has_space: bool,
    ix: PosVec,
    ptr: TreePtr<K, V>,
}

impl<K, V> Position<K, V> {
    fn new() -> Self {
        Self {
            key_found: false,
            leaf_has_space: false,
            ix: PosVec::new(),
            ptr: TreePtr::None,
        }
    }
}

/// Vacant [Entry].
pub struct VacantEntry<'a, K, V> {
    map: &'a mut BTreeMap<K, V>,
    key: K,
    pos: Position<K, V>,
}

impl<'a, K, V> VacantEntry<'a, K, V>
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
        if self.pos.leaf_has_space {
            let result = match self.pos.ptr {
                TreePtr::L(ptr, ix) => unsafe {
                    let x = &mut (*ptr).0;
                    x.insert(ix, (self.key, value));
                    &mut x.ixm(ix).1
                },
                _ => panic!(),
            };
            self.map.len += 1;
            result
        } else {
            &mut self.map.ins_pos(&mut self.pos, self.key, value).1
        }
    }
}

enum OccupiedEntryKey<K, V> {
    First,
    Last,
    Some(Position<K, V>),
}

/// Occupied [Entry].
pub struct OccupiedEntry<'a, K, V> {
    map: &'a mut BTreeMap<K, V>,
    key: OccupiedEntryKey<K, V>,
}

impl<'a, K, V> OccupiedEntry<'a, K, V>
where
    K: Ord,
{
    /// Get reference to entry key.
    pub fn key(&self) -> &K {
        match &self.key {
            OccupiedEntryKey::Some(pos) => pos.ptr.key_ref(),
            OccupiedEntryKey::First => self.map.first_key_value().unwrap().0,
            OccupiedEntryKey::Last => self.map.last_key_value().unwrap().0,
        }
    }

    /// Remove (key,value) from map, returning key and value.
    pub fn remove_entry(self) -> (K, V) {
        match &self.key {
            OccupiedEntryKey::Some(pos) => {
                let result = match pos.ptr {
                    TreePtr::L(ptr, ix) => unsafe { (*ptr).0.remove(ix) },
                    TreePtr::NL(ptr, ix) => unsafe { (*ptr).remove_at(ix) },
                    TreePtr::None => panic!(),
                };
                self.map.len -= 1;
                result
            }
            OccupiedEntryKey::First => self.map.pop_first().unwrap(),
            OccupiedEntryKey::Last => self.map.pop_last().unwrap(),
        }
    }

    /// Remove (key,value) from map, returning the value.
    pub fn remove(self) -> V {
        self.remove_entry().1
    }

    /// Get reference to the value.
    pub fn get(&self) -> &V {
        match &self.key {
            OccupiedEntryKey::Some(pos) => pos.ptr.value_ref(),
            OccupiedEntryKey::First => self.map.first_key_value().unwrap().1,
            OccupiedEntryKey::Last => self.map.last_key_value().unwrap().1,
        }
    }

    /// Get mutable reference to the value.
    pub fn get_mut(&mut self) -> &mut V {
        match &mut self.key {
            OccupiedEntryKey::Some(pos) => pos.ptr.value_mut(),
            OccupiedEntryKey::First => self.map.first_key_value_mut().unwrap().1,
            OccupiedEntryKey::Last => self.map.last_key_value_mut().unwrap().1,
        }
    }

    /// Get mutable reference to the value, consuming the entry.
    pub fn into_mut(mut self) -> &'a mut V {
        match &mut self.key {
            OccupiedEntryKey::Some(pos) => match pos.ptr {
                TreePtr::None => panic!(),
                TreePtr::L(ptr, ix) => unsafe { &mut (*ptr).0.ixm(ix).1 },
                TreePtr::NL(ptr, ix) => unsafe { &mut (*ptr).v.ixm(ix).1 },
            },
            OccupiedEntryKey::First => self.map.first_key_value_mut().unwrap().1,
            OccupiedEntryKey::Last => self.map.last_key_value_mut().unwrap().1,
        }
    }

    /// Update the value returns the old value.
    pub fn insert(&mut self, value: V) -> V {
        std::mem::replace(self.get_mut(), value)
    }
}

enum Tree<K, V> {
    L(Leaf<K, V>),
    NL(NonLeaf<K, V>),
}

impl<K, V> Default for Tree<K, V> {
    fn default() -> Self {
        Tree::L(Leaf(FixedCapVec::new()))
    }
}

impl<K, V> Tree<K, V> {
    fn insert(&mut self, key: K, x: &mut InsertCtx<K, V>)
    where
        K: Ord,
    {
        match self {
            Tree::L(leaf) => leaf.insert(key, x),
            Tree::NL(nonleaf) => nonleaf.insert(key, x),
        }
    }

    fn prepare_insert(&mut self, pos: &mut PosVec, level: usize) -> Option<Split<K, V>> {
        match self {
            Tree::L(leaf) => leaf.prepare_insert(pos),
            Tree::NL(nonleaf) => nonleaf.prepare_insert(pos, level),
        }
    }

    fn do_insert(&mut self, pos: &[u8], level: usize, key: K, value: V) -> &mut (K, V) {
        match self {
            Tree::L(leaf) => leaf.do_insert(pos, level, key, value),
            Tree::NL(nonleaf) => nonleaf.do_insert(pos, level, key, value),
        }
    }

    fn new_root(&mut self, (med, right): Split<K, V>) {
        let left = std::mem::take(self);
        let mut v = FixedCapVec::new();
        v.push(med);
        let mut c = FixedCapVec::new();
        c.push(left);
        c.push(right);
        *self = Tree::NL(NonLeaf { v, c });
    }

    fn remove<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self {
            Tree::L(leaf) => leaf.remove(key),
            Tree::NL(nonleaf) => nonleaf.remove(key),
        }
    }

    fn find_position<Q>(&mut self, key: &Q, pos: &mut Position<K, V>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self {
            Tree::L(leaf) => leaf.find_position(key, pos),
            Tree::NL(nonleaf) => nonleaf.find_position(key, pos),
        }
    }

    fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self {
            Tree::L(leaf) => leaf.get_key_value(key),
            Tree::NL(nonleaf) => nonleaf.get_key_value(key),
        }
    }

    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut (K, V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self {
            Tree::L(leaf) => leaf.get_mut(key),
            Tree::NL(nonleaf) => nonleaf.get_mut(key),
        }
    }

    fn pop_first(&mut self) -> Option<(K, V)> {
        match self {
            Tree::L(leaf) => leaf.pop_first(),
            Tree::NL(nonleaf) => nonleaf.pop_first(),
        }
    }

    fn pop_last(&mut self) -> Option<(K, V)> {
        match self {
            Tree::L(leaf) => leaf.0.pop(),
            Tree::NL(nonleaf) => nonleaf.pop_last(),
        }
    }

    fn retain<F>(&mut self, f: &mut F) -> usize
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        match self {
            Tree::L(leaf) => leaf.retain(f),
            Tree::NL(nonleaf) => nonleaf.retain(f),
        }
    }

    fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut(match self {
            Tree::L(x) => IterMutE::L(x.iter_mut()),
            Tree::NL(x) => IterMutE::NL(Box::new(x.iter_mut())),
        })
    }

    fn iter(&self) -> Iter<'_, K, V> {
        Iter(match self {
            Tree::L(x) => IterE::L(x.iter()),
            Tree::NL(x) => IterE::NL(Box::new(x.iter())),
        })
    }

    fn range_mut<T, R>(&mut self, range: &R, left: bool, right: bool) -> IterMut<'_, K, V>
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        IterMut(match self {
            Tree::L(x) => IterMutE::L(x.range_mut(range)),
            Tree::NL(x) => IterMutE::NL(Box::new(x.range_mut(range, left, right))),
        })
    }

    fn range<T, R>(&self, range: &R, left: bool, right: bool) -> Iter<'_, K, V>
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        Iter(match self {
            Tree::L(x) => IterE::L(x.range(range)),
            Tree::NL(x) => IterE::NL(Box::new(x.range(range, left, right))),
        })
    }

    fn walk<F, Q>(&self, start: &Q, action: &mut F) -> bool
    where
        F: FnMut(&(K, V)) -> bool,
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self {
            Tree::L(leaf) => {
                for i in leaf.skip(start)..leaf.0.len() {
                    if action(leaf.0.ix(i)) {
                        return true;
                    }
                }
            }
            Tree::NL(nonleaf) => {
                let i = nonleaf.skip(start);
                if nonleaf.c.ix(i).walk(start, action) {
                    return true;
                }
                for i in i..nonleaf.v.len() {
                    let v = nonleaf.v.ix(i);
                    if start <= v.0.borrow() && action(v) {
                        return true;
                    }
                    if nonleaf.c.ix(i + 1).walk(start, action) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn walk_mut<F, Q>(&mut self, start: &Q, action: &mut F) -> bool
    where
        F: FnMut(&mut (K, V)) -> bool,
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self {
            Tree::L(leaf) => {
                for i in leaf.skip(start)..leaf.0.len() {
                    if action(leaf.0.ixm(i)) {
                        return true;
                    }
                }
            }
            Tree::NL(nonleaf) => {
                let i = nonleaf.skip(start);
                if i < nonleaf.c.len() && nonleaf.c.ixm(i).walk_mut(start, action) {
                    return true;
                }
                for i in i..nonleaf.v.len() {
                    let v = nonleaf.v.ixm(i);
                    if start <= v.0.borrow() && action(v) {
                        return true;
                    }
                    if nonleaf.c.ixm(i + 1).walk_mut(start, action) {
                        return true;
                    }
                }
            }
        }
        false
    }
} // End impl Tree

struct Leaf<K, V>(LeafVec<K, V>);

impl<K, V> Leaf<K, V> {
    fn full(&self) -> bool {
        self.0.len() >= LEAF_FULL
    }

    fn split(&mut self) -> ((K, V), LeafVec<K, V>) {
        let right = self.0.split_off(LEAF_SPLIT);
        let med = self.0.pop().unwrap();
        (med, right)
    }

    fn insert(&mut self, key: K, x: &mut InsertCtx<K, V>)
    where
        K: Ord,
    {
        let mut i = match self.0.search(|x| x.0.borrow().cmp(&key)) {
            Ok(i) => {
                let value = x.value.take().unwrap();
                x.value = Some(std::mem::replace(self.0.ixm(i), (key, value)).1);
                return;
            }
            Err(i) => i,
        };
        let value = x.value.take().unwrap();
        if self.full() {
            let (med, mut right) = self.split();
            if i >= LEAF_SPLIT {
                i -= LEAF_SPLIT;
                right.insert(i, (key, value));
            } else {
                self.0.insert(i, (key, value));
            }
            let right = Tree::L(Self(right));
            x.split = Some((med, right));
        } else {
            self.0.insert(i, (key, value));
        }
    }

    fn prepare_insert(&mut self, pos: &mut PosVec) -> Option<Split<K, V>> {
        debug_assert!(self.full());
        let mut level = pos.len() - 1;
        if level == 0 {
            level += 1;
            pos.insert(0, 0);
        }
        if pos[level] >= LEAF_SPLIT as u8 {
            pos[level] -= LEAF_SPLIT as u8;
            pos[level - 1] += 1;
        }
        let (med, right) = self.split();
        let right = Tree::L(Self(right));
        Some((med, right))
    }

    fn skip<Q>(&self, key: &Q) -> usize
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self.0.search(|x| x.0.borrow().cmp(key)) {
            Ok(i) => i,
            Err(i) => i,
        }
    }

    fn find_position<Q>(&mut self, key: &Q, pos: &mut Position<K, V>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let i = match self.0.search(|x| x.0.borrow().cmp(key)) {
            Ok(i) => {
                pos.key_found = true;
                pos.ptr = TreePtr::L(self, i);
                i
            }
            Err(i) => i,
        };
        pos.ix.push(i as u8);
        if !self.full() {
            pos.leaf_has_space = true;
            pos.ptr = TreePtr::L(self, i);
        }
    }

    fn do_insert(&mut self, pos: &[u8], level: usize, key: K, value: V) -> &mut (K, V) {
        let i = pos[level] as usize;
        self.0.insert(i, (key, value));
        self.0.ixm(i)
    }

    fn remove<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self.0.search(|x| x.0.borrow().cmp(key)) {
            Ok(i) => Some(self.0.remove(i)),
            Err(_i) => None,
        }
    }

    fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self.0.search(|x| x.0.borrow().cmp(key)) {
            Ok(i) => {
                let x = self.0.ix(i);
                Some((&x.0, &x.1))
            }
            Err(_i) => None,
        }
    }

    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut (K, V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self.0.search(|x| x.0.borrow().cmp(key)) {
            Ok(i) => Some(self.0.ixm(i)),
            Err(_i) => None,
        }
    }

    fn pop_first(&mut self) -> Option<(K, V)> {
        if self.0.is_empty() {
            return None;
        }
        Some(self.0.remove(0))
    }

    fn retain<F>(&mut self, f: &mut F) -> usize
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        let mut removed = 0;
        self.0.retain_mut(|(k, v)| {
            let ok = f(k, v);
            if !ok {
                removed += 1
            };
            ok
        });
        removed
    }

    fn iter_mut(&mut self) -> IterLeafMut<'_, K, V> {
        IterLeafMut(self.0.iter_mut())
    }

    fn iter(&self) -> IterLeaf<'_, K, V> {
        IterLeaf(self.0.iter())
    }

    fn range_mut<T, R>(&mut self, range: &R) -> IterLeafMut<'_, K, V>
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        let mut x = 0;
        while x < self.0.len() && !range.contains(self.0.ix(x).0.borrow()) {
            x += 1;
        }
        let mut y = self.0.len();
        while y > x && !range.contains(self.0.ix(y - 1).0.borrow()) {
            y -= 1;
        }
        IterLeafMut(self.0[x..y].iter_mut())
    }

    fn range<T, R>(&self, range: &R) -> IterLeaf<'_, K, V>
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        let mut x = 0;
        while x < self.0.len() && !range.contains(self.0[x].0.borrow()) {
            x += 1;
        }
        let mut y = self.0.len();
        while y > x && !range.contains(self.0.ix(y - 1).0.borrow()) {
            y -= 1;
        }
        IterLeaf(self.0[x..y].iter())
    }
} // End impl Leaf

struct NonLeaf<K, V> {
    v: NonLeafVec<K, V>,
    c: NonLeafChildVec<K, V>,
}

impl<K, V> NonLeaf<K, V> {
    fn full(&self) -> bool {
        self.v.len() == NON_LEAF_FULL
    }

    fn skip<Q>(&self, key: &Q) -> usize
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self.v.search(|x| x.0.borrow().cmp(key)) {
            Ok(i) => i,
            Err(i) => i,
        }
    }

    fn remove_at(&mut self, i: usize) -> (K, V) {
        if let Some(x) = self.c.ixm(i).pop_last() {
            std::mem::replace(self.v.ixm(i), x)
        } else {
            self.c.remove(i);
            self.v.remove(i)
        }
    }

    fn find_position<Q>(&mut self, key: &Q, pos: &mut Position<K, V>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self.v.search(|x| x.0.borrow().cmp(key)) {
            Ok(i) => {
                pos.key_found = true;
                pos.ptr = TreePtr::NL(self, i);
                pos.ix.push(i as u8);
            }
            Err(i) => {
                pos.ix.push(i as u8);
                self.c.ixm(i).find_position(key, pos);
            }
        }
    }

    fn split(&mut self) -> Split<K, V> {
        let right = Self {
            v: self.v.split_off(NON_LEAF_SPLIT),
            c: self.c.split_off(NON_LEAF_SPLIT),
        };
        let med = self.v.pop().unwrap();
        (med, Tree::NL(right))
    }

    fn insert(&mut self, key: K, x: &mut InsertCtx<K, V>)
    where
        K: Ord,
    {
        match self.v.search(|x| x.0.borrow().cmp(&key)) {
            Ok(i) => {
                let value = x.value.take().unwrap();
                x.value = Some(std::mem::replace(self.v.ixm(i), (key, value)).1);
            }
            Err(i) => {
                self.c.ixm(i).insert(key, x);
                if let Some((med, right)) = x.split.take() {
                    self.v.insert(i, med);
                    self.c.insert(i + 1, right);
                    if self.full() {
                        x.split = Some(self.split());
                    }
                }
            }
        }
    }

    fn prepare_insert(&mut self, pos: &mut PosVec, mut level: usize) -> Option<Split<K, V>> {
        let i = pos[level] as usize;
        if let Some((med, right)) = self.c.ixm(i).prepare_insert(pos, level + 1) {
            self.v.insert(i, med);
            self.c.insert(i + 1, right);
        }
        if self.full() {
            if level == 0 {
                pos.insert(0, 0);
                level += 1;
            }
            if pos[level] >= NON_LEAF_SPLIT as u8 {
                pos[level] -= NON_LEAF_SPLIT as u8;
                pos[level - 1] += 1;
            }
            Some(self.split())
        } else {
            None
        }
    }

    fn do_insert(&mut self, pos: &[u8], level: usize, key: K, value: V) -> &mut (K, V) {
        let i = pos[level] as usize;
        self.c.ixm(i).do_insert(pos, level + 1, key, value)
    }

    fn remove<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self.v.search(|x| x.0.borrow().cmp(key)) {
            Ok(i) => Some(self.remove_at(i)),
            Err(i) => self.c.ixm(i).remove(key),
        }
    }

    fn retain<F>(&mut self, f: &mut F) -> usize
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        let mut removed = 0;
        let mut i = 0;
        while i < self.v.len() {
            removed += self.c.ixm(i).retain(f);
            let e = self.v.ixm(i);
            if !f(&e.0, &mut e.1) {
                removed += 1;
                if let Some(x) = self.c.ixm(i).pop_last() {
                    let _ = std::mem::replace(self.v.ixm(i), x);
                    i += 1;
                } else {
                    self.c.remove(i);
                    self.v.remove(i);
                }
            } else {
                i += 1;
            }
        }
        removed += self.c.ixm(i).retain(f);
        removed
    }

    fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let mut i = 0;
        while i < self.v.len() {
            match self.v.ix(i).0.borrow().cmp(key) {
                Ordering::Equal => {
                    let kv = &self.v.ix(i);
                    return Some((&kv.0, &kv.1));
                }
                Ordering::Greater => {
                    return self.c.ix(i).get_key_value(key);
                }
                Ordering::Less => {
                    i += 1;
                }
            }
        }
        self.c.ix(i).get_key_value(key)
    }

    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut (K, V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let mut i = 0;
        while i < self.v.len() {
            match self.v.ix(i).0.borrow().cmp(key) {
                Ordering::Equal => {
                    return Some(self.v.ixm(i));
                }
                Ordering::Greater => {
                    return self.c.ixm(i).get_mut(key);
                }
                Ordering::Less => {
                    i += 1;
                }
            }
        }
        self.c.ixm(i).get_mut(key)
    }

    fn pop_first(&mut self) -> Option<(K, V)> {
        if let Some(x) = self.c.ixm(0).pop_first() {
            Some(x)
        } else if self.v.is_empty() {
            None
        } else {
            self.c.remove(0);
            Some(self.v.remove(0))
        }
    }

    fn pop_last(&mut self) -> Option<(K, V)> {
        let i = self.c.len();
        if let Some(x) = self.c.ixm(i - 1).pop_last() {
            Some(x)
        } else if self.v.is_empty() {
            None
        } else {
            self.c.pop();
            self.v.pop()
        }
    }

    fn iter_mut(&mut self) -> IterNonLeafMut<'_, K, V> {
        let (v, c) = (self.v.iter_mut(), self.c.iter_mut());
        IterNonLeafMut {
            v,
            c,
            current: None,
            current_back: None,
        }
    }

    fn iter(&self) -> IterNonLeaf<'_, K, V> {
        let (v, c) = (self.v.iter(), self.c.iter());
        IterNonLeaf {
            v,
            c,
            current: None,
            current_back: None,
        }
    }

    fn get_xy<T, R>(&self, range: &R) -> (usize, usize)
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        let (mut x, b) = (0, range.start_bound());
        while x < self.v.len() {
            match b {
                Bound::Included(start) => {
                    if self.v[x].0.borrow() >= start {
                        break;
                    }
                }
                Bound::Excluded(start) => {
                    if self.v[x].0.borrow() > start {
                        break;
                    }
                }
                Bound::Unbounded => break,
            }
            x += 1;
        }
        let (mut y, b) = (self.v.len(), range.end_bound());
        while y > x {
            match b {
                Bound::Included(end) => {
                    if self.v[y - 1].0.borrow() <= end {
                        break;
                    }
                }
                Bound::Excluded(end) => {
                    if self.v[y - 1].0.borrow() < end {
                        break;
                    }
                }
                Bound::Unbounded => break,
            }
            y -= 1;
        }
        (x, y)
    }

    fn range_mut<T, R>(&mut self, range: &R, left: bool, right: bool) -> IterNonLeafMut<'_, K, V>
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        let (x, y) = self.get_xy(range);
        let (v, mut c) = (self.v[x..y].iter_mut(), self.c[x..y + 1].iter_mut());
        let current = if left || x == y {
            let tree = c.next().unwrap();
            Some(tree.range_mut(range, left, right && x == y))
        } else {
            None
        };
        let current_back = if right && x != y {
            let tree = c.next_back().unwrap();
            Some(tree.range_mut(range, false, true))
        } else {
            None
        };
        IterNonLeafMut {
            v,
            c,
            current,
            current_back,
        }
    }

    fn range<T, R>(&self, range: &R, left: bool, right: bool) -> IterNonLeaf<'_, K, V>
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        let (x, y) = self.get_xy(range);
        let (v, mut c) = (self.v[x..y].iter(), self.c[x..y + 1].iter());
        let current = if left || x == y {
            let tree = c.next().unwrap();
            Some(tree.range(range, left, right && x == y))
        } else {
            None
        };
        let current_back = if right && x != y {
            let tree = c.next_back().unwrap();
            Some(tree.range(range, false, true))
        } else {
            None
        };
        IterNonLeaf {
            v,
            c,
            current,
            current_back,
        }
    }
} // End impl NonLeaf

/// Mutable iterator returned by [BTreeMap::iter_mut], [BTreeMap::range_mut].
pub struct IterMut<'a, K, V>(IterMutE<'a, K, V>);

enum IterMutE<'a, K, V> {
    L(IterLeafMut<'a, K, V>),
    NL(Box<IterNonLeafMut<'a, K, V>>),
    Empty,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a mut K, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            IterMutE::L(x) => x.next(),
            IterMutE::NL(x) => x.next(),
            IterMutE::Empty => None,
        }
    }
}
impl<'a, K, V> DoubleEndedIterator for IterMut<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            IterMutE::L(x) => x.next_back(),
            IterMutE::NL(x) => x.next_back(),
            IterMutE::Empty => None,
        }
    }
}
impl<'a, K, V> FusedIterator for IterMut<'a, K, V> {}

/// Iterator returned by [BTreeMap::iter], [BTreeMap::range].
pub struct Iter<'a, K, V>(IterE<'a, K, V>);

enum IterE<'a, K, V> {
    L(IterLeaf<'a, K, V>),
    NL(Box<IterNonLeaf<'a, K, V>>),
    Empty,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            IterE::L(x) => x.next(),
            IterE::NL(x) => x.next(),
            IterE::Empty => None,
        }
    }
}
impl<'a, K, V> DoubleEndedIterator for Iter<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            IterE::L(x) => x.next_back(),
            IterE::NL(x) => x.next_back(),
            IterE::Empty => None,
        }
    }
}
impl<'a, K, V> FusedIterator for Iter<'a, K, V> {}

/// Consuming iterator, result of converting BTreeMap into an iterator.
pub struct IntoIter<K, V>(BTreeMap<K, V>);

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop_first()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.0.len, Some(self.0.len))
    }
}
impl<K, V> DoubleEndedIterator for IntoIter<K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.pop_last()
    }
}
impl<K, V> FusedIterator for IntoIter<K, V> {}

/// Consuming iterator returned by [BTreeMap::into_keys].
pub struct IntoKeys<K, V>(BTreeMap<K, V>);

impl<K, V> Iterator for IntoKeys<K, V> {
    type Item = K;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.0.pop_first()?.0)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.0.len, Some(self.0.len))
    }
}
impl<K, V> DoubleEndedIterator for IntoKeys<K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(self.0.pop_last()?.0)
    }
}
impl<K, V> FusedIterator for IntoKeys<K, V> {}

/// Consuming iterator returned by [BTreeMap::into_values].
pub struct IntoValues<K, V>(BTreeMap<K, V>);

impl<K, V> Iterator for IntoValues<K, V> {
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.0.pop_first()?.1)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.0.len, Some(self.0.len))
    }
}
impl<K, V> DoubleEndedIterator for IntoValues<K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(self.0.pop_last()?.1)
    }
}
impl<K, V> FusedIterator for IntoValues<K, V> {}

// Leaf iterators.

struct IterLeafMut<'a, K, V>(std::slice::IterMut<'a, (K, V)>);

impl<'a, K, V> Iterator for IterLeafMut<'a, K, V> {
    type Item = (&'a mut K, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        let &mut (ref mut k, ref mut v) = self.0.next()?;
        Some((k, v))
    }
}
impl<'a, K, V> DoubleEndedIterator for IterLeafMut<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let &mut (ref mut k, ref mut v) = self.0.next_back()?;
        Some((k, v))
    }
}

struct IterLeaf<'a, K, V>(std::slice::Iter<'a, (K, V)>);

impl<'a, K, V> Iterator for IterLeaf<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        let (k, v) = self.0.next()?;
        Some((k, v))
    }
}
impl<'a, K, V> DoubleEndedIterator for IterLeaf<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let (k, v) = self.0.next_back()?;
        Some((k, v))
    }
}

// Non-Leaf iterators.

struct IterNonLeafMut<'a, K, V> {
    v: std::slice::IterMut<'a, (K, V)>,
    c: std::slice::IterMut<'a, Tree<K, V>>,
    current: Option<IterMut<'a, K, V>>,
    current_back: Option<IterMut<'a, K, V>>,
}
impl<'a, K, V> IterNonLeafMut<'a, K, V> {
    fn current(&mut self) -> &mut IterMut<'a, K, V> {
        if self.current.is_none() {
            self.current = Some(if let Some(tree) = self.c.next() {
                tree.iter_mut()
            } else {
                IterMut(IterMutE::Empty)
            });
        }
        self.current.as_mut().unwrap()
    }

    fn current_back(&mut self) -> &mut IterMut<'a, K, V> {
        if self.current_back.is_none() {
            self.current_back = Some(if let Some(tree) = self.c.next_back() {
                tree.iter_mut()
            } else {
                IterMut(IterMutE::Empty)
            });
        }
        self.current_back.as_mut().unwrap()
    }
}

impl<'a, K, V> Iterator for IterNonLeafMut<'a, K, V> {
    type Item = (&'a mut K, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(x) = self.current().next() {
            Some(x)
        } else if let Some(&mut (ref mut k, ref mut v)) = self.v.next() {
            self.current = None;
            Some((k, v))
        } else {
            self.current_back().next()
        }
    }
}
impl<'a, K, V> DoubleEndedIterator for IterNonLeafMut<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if let Some(x) = self.current_back().next_back() {
            Some(x)
        } else if let Some(&mut (ref mut k, ref mut v)) = self.v.next_back() {
            self.current_back = None;
            Some((k, v))
        } else {
            self.current().next_back()
        }
    }
}

struct IterNonLeaf<'a, K, V> {
    v: std::slice::Iter<'a, (K, V)>,
    c: std::slice::Iter<'a, Tree<K, V>>,
    current: Option<Iter<'a, K, V>>,
    current_back: Option<Iter<'a, K, V>>,
}
impl<'a, K, V> IterNonLeaf<'a, K, V> {
    fn current(&mut self) -> &mut Iter<'a, K, V> {
        if self.current.is_none() {
            self.current = Some(if let Some(tree) = self.c.next() {
                tree.iter()
            } else {
                Iter(IterE::Empty)
            });
        }
        self.current.as_mut().unwrap()
    }

    fn current_back(&mut self) -> &mut Iter<'a, K, V> {
        if self.current_back.is_none() {
            self.current_back = Some(if let Some(tree) = self.c.next_back() {
                tree.iter()
            } else {
                Iter(IterE::Empty)
            });
        }
        self.current_back.as_mut().unwrap()
    }
}
impl<'a, K, V> Iterator for IterNonLeaf<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(x) = self.current().next() {
            Some(x)
        } else if let Some((k, v)) = self.v.next() {
            self.current = None;
            Some((k, v))
        } else {
            self.current_back().next()
        }
    }
}
impl<'a, K, V> DoubleEndedIterator for IterNonLeaf<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if let Some(x) = self.current_back().next_back() {
            Some(x)
        } else if let Some((k, v)) = self.v.next_back() {
            self.current_back = None;
            Some((k, v))
        } else {
            self.current().next_back()
        }
    }
}

// Trivial iterators.

/// Iterator returned by [BTreeMap::values_mut].
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
impl<'a, K, V> FusedIterator for ValuesMut<'a, K, V> {}

/// Mutable iterator returned by [BTreeMap::values].
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
impl<'a, K, V> FusedIterator for Values<'a, K, V> {}

/// Iterator returned by [BTreeMap::keys].
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
impl<'a, K, V> FusedIterator for Keys<'a, K, V> {}

#[test]
fn basic_range_test() {
    let mut map = BTreeMap::<usize, usize>::new();
    for i in 0..100 {
        map.insert(i, i);
    }

    for j in 0..100 {
        assert_eq!(map.range(0..=j).count(), j + 1);
    }
}

#[test]
fn test_exp_insert() {
    for _rep in 0..1000 {
        let mut t = /*std::collections::*/ BTreeMap::<usize, usize>::default();
        let n = 10000;
        for i in 0..n {
            t.insert(i, i);
        }
    }
}

#[test]
fn test_std_insert() {
    for _rep in 0..1000 {
        let mut t = std::collections::BTreeMap::<usize, usize>::default();
        let n = 10000;
        for i in 0..n {
            t.insert(i, i);
        }
    }
}

#[test]
fn test_exp_entry() {
    for _rep in 0..1000 {
        let mut t = /*std::collections::*/ BTreeMap::<usize, usize>::default();
        let n = 10000;
        for i in 0..n {
            t.entry(i).or_insert(i);
        }
    }
}

#[test]
fn test_std_entry() {
    for _rep in 0..1000 {
        let mut t = std::collections::BTreeMap::<usize, usize>::default();
        let n = 10000;
        for i in 0..n {
            t.entry(i).or_insert(i);
        }
    }
}

#[test]
fn various_tests() {
    for _rep in 0..1000 {
        let mut t = /*std::collections::*/ BTreeMap::<usize, usize>::default();
        let n = 10000;
        for i in 0..n {
            t.insert(i, i);
        }
        if true {
            assert!(t.first_key_value().unwrap().0 == &0);
            assert!(t.last_key_value().unwrap().0 == &(n - 1));

            println!("doing for x in & test");
            for x in &t {
                if *x.0 < 50 {
                    print!("{:?};", x);
                }
            }
            println!();

            println!("doing for x in &mut test");
            for x in &mut t {
                *x.1 *= 1;
                if *x.0 < 50 {
                    print!("{:?};", x);
                }
            }
            println!();

            println!("doing range mut test");

            for x in t.range_mut(20..=60000).rev() {
                if *x.0 < 50 {
                    print!("{:?};", x);
                }
            }
            println!("done range mut test");

            println!("t.len()={} doing range non-mut test", t.len());

            for x in t.range(20..=60000).rev() {
                if *x.0 < 50 {
                    print!("{:?};", x);
                }
            }
            println!("done range non-mut test");

            println!("doing get test");
            for i in 0..n {
                assert_eq!(t.get(&i).unwrap(), &i);
            }

            println!("doing get_mut test");
            for i in 0..n {
                assert_eq!(t.get_mut(&i).unwrap(), &i);
            }

            /*
                    println!("t.len()={} doing walk test", t.len());
                    t.walk(&10, &mut |(k, _): &(usize, usize)| {
                        if *k <= 50 {
                            print!("{:?};", k);
                            false
                        } else {
                            true
                        }
                    });
                    println!();
            */

            println!("doing remove evens test");
            for i in 0..n {
                if i % 2 == 0 {
                    assert_eq!(t.remove(&i).unwrap(), i);
                }
            }

            /*
                    println!("t.len()={} re-doing walk test", t.len());
                    t.walk(&10, &mut |(k, _): &(usize, usize)| {
                        if *k <= 50 {
                            print!("{:?};", k);
                            false
                        } else {
                            true
                        }
                    });
                    println!();
            */

            println!("doing retain test - retain only keys divisible by 5");
            t.retain(|k, _v| k % 5 == 0);

            println!("Consuming iterator test");
            for x in t {
                if x.0 < 50 {
                    print!("{:?};", x);
                }
            }
            println!();

            println!("FromIter collect test");
            let a = [1, 2, 3];
            let map: BTreeMap<i32, i32> = a.iter().map(|&x| (x, x * x)).collect();
            for x in map {
                print!("{:?};", x);
            }
            println!();

            println!("From test");
            let map = BTreeMap::from([(1, 2), (3, 4)]);
            for x in map {
                print!("{:?};", x);
            }
            println!();
        }
    }
}

#[cfg(test)]
mod tests;
