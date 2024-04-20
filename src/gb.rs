// Vector types.
use crate::vecs;
use arrayvec::ArrayVec;
use std::{
    borrow::Borrow,
    cmp::Ordering,
    fmt,
    fmt::Debug,
    iter::FusedIterator,
    marker::PhantomData,
    ops::{Bound, RangeBounds},
};
use vecs::{FixedCapIter, FixedCapVec};

type LeafVec<K, V, const B: usize> = FixedCapVec<(K, V), B>;
type NonLeafVec<K, V, const B: usize> = FixedCapVec<(K, V), B>;
type NonLeafChildVec<K, V, const B: usize> = FixedCapVec<Tree<K, V, B>, B>;

const AX: usize = 10; // Size for fixed ArrayVecs, 10 should probably be enough.

type PosVec = ArrayVec<u8, AX>;
type StkMutVec<'a, K, V, const B: usize> = ArrayVec<StkMut<'a, K, V, B>, AX>;
type StkConVec<K, V, const B: usize> = ArrayVec<StkCon<K, V, B>, AX>;
type StkVec<'a, K, V, const B: usize> = ArrayVec<Stk<'a, K, V, B>, AX>;

type Split<K, V, const B: usize> = ((K, V), Tree<K, V, B>);

fn check_range<T, R>(range: &R)
where
    T: Ord + ?Sized,
    R: RangeBounds<T>,
{
    use Bound::*;
    match (range.start_bound(), range.end_bound()) {
        (Included(s), Included(e)) => {
            if e < s {
                panic!("range start is greater than range end in BTreeMap")
            }
        }
        (Included(s), Excluded(e)) => {
            if e < s {
                panic!("range start is greater than range end in BTreeMap")
            }
        }
        (Excluded(s), Included(e)) => {
            if e < s {
                panic!("range start is greater than range end in BTreeMap")
            }
        }
        (Excluded(s), Excluded(e)) => {
            if e == s {
                panic!("range start and end are equal and excluded in BTreeMap")
            }
            if e < s {
                panic!("range start is greater than range end in BTreeMap")
            }
        }
        _ => {}
    }
}

/// BTreeMap similar to [std::collections::BTreeMap] where B value can be specified.
/// B should be an odd number, at least 11, a good value may be 39. Must be less than 256.
pub struct BTreeMap<K, V, const B: usize> {
    len: usize,
    tree: Tree<K, V, B>,
}
impl<K, V, const B: usize> Default for BTreeMap<K, V, B> {
    fn default() -> Self {
        Self::new()
    }
}
impl<K, V, const B: usize> BTreeMap<K, V, B> {
    #[cfg(test)]
    pub(crate) fn check(&self) {}

    const CHECK_B: usize = {
        assert!(B >= 13 && B <= 255);
        0
    };

    /// Returns a new, empty map.
    pub fn new() -> Self {
        Self {
            len: Self::CHECK_B,
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
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, B>
    where
        K: Ord,
    {
        let mut pos = Position::new();
        self.tree.find_position(&key, &mut pos);
        if pos.key_found {
            let key = OccupiedEntryKey::Some(pos);
            Entry::Occupied(OccupiedEntry {
                map: self,
                key,
                _pd: PhantomData,
            })
        } else {
            Entry::Vacant(VacantEntry {
                map: self,
                key,
                pos,
                _pd: PhantomData,
            })
        }
    }

    /// Get first Entry.
    pub fn first_entry(&mut self) -> Option<OccupiedEntry<'_, K, V, B>> {
        if self.is_empty() {
            None
        } else {
            Some(OccupiedEntry {
                map: self,
                key: OccupiedEntryKey::First,
                _pd: PhantomData,
            })
        }
    }

    /// Get last Entry.
    pub fn last_entry(&mut self) -> Option<OccupiedEntry<'_, K, V, B>> {
        if self.is_empty() {
            None
        } else {
            Some(OccupiedEntry {
                map: self,
                key: OccupiedEntryKey::Last,
                _pd: PhantomData,
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

    /// Tries to insert a key-value pair into the map, and returns
    /// a mutable reference to the value in the entry.
    ///
    /// If the map already had this key present, nothing is updated, and
    /// an error containing the occupied entry and the value is returned.
    pub fn try_insert(&mut self, key: K, value: V) -> Result<&mut V, OccupiedError<'_, K, V, B>>
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

    /// Get references to first key and value, value reference is mutable.
    fn first_key_value_mut(&mut self) -> Option<(&K, &mut V)> {
        self.tree.iter_mut().next()
    }

    /// Get references to last key and value, value reference is mutable.
    fn last_key_value_mut(&mut self) -> Option<(&K, &mut V)> {
        self.tree.iter_mut().next_back()
    }

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    ///
    /// If a key from `other` is already present in `self`, the respective
    /// value from `self` will be overwritten with the respective value from `other`.
    pub fn append(&mut self, other: &mut BTreeMap<K, V, B>)
    where
        K: Ord,
    {
        let (tree, len) = (std::mem::take(&mut other.tree), other.len);
        other.len = 0;
        let temp = BTreeMap { len, tree };
        // Could use Cursor to do this more efficiently.
        for (k, v) in temp {
            self.insert(k, v);
        }
    }

    /// Splits the collection into two at the given key.
    /// Returns everything after the given key, including the key.
    pub fn split_off<Q: ?Sized + Ord>(&mut self, key: &Q) -> Self
    where
        K: Borrow<Q> + Ord,
    {
        let mut map = Self::new();
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
    pub fn extract_if<F>(&mut self, pred: F) -> ExtractIf<'_, K, V, B, F>
    where
        K: Ord,
        F: FnMut(&K, &mut V) -> bool,
    {
        let source = self.lower_bound_mut(Bound::Unbounded);
        ExtractIf { source, pred }
    }

    /// Get iterator of references to key-value pairs.
    pub fn iter(&self) -> Iter<'_, K, V, B> {
        Iter {
            len: self.len,
            inner: self.tree.iter(),
        }
    }

    /// Get iterator of mutable references to key-value pairs.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V, B> {
        IterMut {
            len: self.len,
            inner: self.tree.iter_mut(),
        }
    }

    /// Get iterator for range of references to key-value pairs.
    pub fn range<T, R>(&self, range: R) -> Range<'_, K, V, B>
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
    pub fn range_mut<T, R>(&mut self, range: R) -> RangeMut<'_, K, V, B>
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        check_range(&range);
        self.tree.range_mut(&range)
    }

    /// Get iterator of references to keys.
    pub fn keys(&self) -> Keys<'_, K, V, B> {
        Keys(self.iter())
    }

    /// Get iterator of references to values.
    pub fn values(&self) -> Values<'_, K, V, B> {
        Values(self.iter())
    }

    /// Get iterator of mutable references to values.
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V, B> {
        ValuesMut(self.iter_mut())
    }

    /// Get consuming iterator that returns all the keys, in sorted order.
    pub fn into_keys(self) -> IntoKeys<K, V, B> {
        IntoKeys(self.into_iter())
    }

    /// Get consuming iterator that returns all the values, in sorted order.
    pub fn into_values(self) -> IntoValues<K, V, B> {
        IntoValues(self.into_iter())
    }

    /// Get a cursor positioned just after bound that permits map mutation.
    pub fn lower_bound_mut<Q>(&mut self, bound: Bound<&Q>) -> CursorMut<'_, K, V, B>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        CursorMut::lower_bound(self, bound)
    }

    /// Get a cursor positioned just before bound that permits map mutation.
    pub fn upper_bound_mut<Q>(&mut self, bound: Bound<&Q>) -> CursorMut<'_, K, V, B>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        CursorMut::upper_bound(self, bound)
    }

    /// Get cursor positioned just after bound.
    pub fn lower_bound<Q>(&self, bound: Bound<&Q>) -> Cursor<'_, K, V, B>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        Cursor::lower_bound(self, bound)
    }

    /// Get cursor positioned just before bound.
    pub fn upper_bound<Q>(&self, bound: Bound<&Q>) -> Cursor<'_, K, V, B>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        Cursor::upper_bound(self, bound)
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

    fn ins_pos(&mut self, pos: &mut Position<K, V, B>, key: K, value: V) -> &mut (K, V) {
        if let Some(s) = self.tree.prepare_insert(&mut pos.ix, 0) {
            self.tree.new_root(s);
        }
        self.len += 1;
        self.tree.do_insert(&pos.ix, 0, key, value)
    }
} // End impl BTreeMap

use std::hash::{Hash, Hasher};
impl<K: Hash, V: Hash, const B: usize> Hash for BTreeMap<K, V, B> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // state.write_length_prefix(self.len());
        for elt in self.iter() {
            elt.hash(state);
        }
    }
}
impl<K: PartialEq, V: PartialEq, const B: usize> PartialEq for BTreeMap<K, V, B> {
    fn eq(&self, other: &BTreeMap<K, V, B>) -> bool {
        self.len() == other.len() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}
impl<K: Eq, V: Eq, const B: usize> Eq for BTreeMap<K, V, B> {}

impl<K: PartialOrd, V: PartialOrd, const B: usize> PartialOrd for BTreeMap<K, V, B> {
    fn partial_cmp(&self, other: &BTreeMap<K, V, B>) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}
impl<K: Ord, V: Ord, const B: usize> Ord for BTreeMap<K, V, B> {
    fn cmp(&self, other: &BTreeMap<K, V, B>) -> Ordering {
        self.iter().cmp(other.iter())
    }
}
impl<K, V, const B: usize> IntoIterator for BTreeMap<K, V, B> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V, B>;

    /// Convert BTreeMap to Iterator.
    fn into_iter(self) -> IntoIter<K, V, B> {
        IntoIter::new(self)
    }
}
impl<'a, K, V, const B: usize> IntoIterator for &'a BTreeMap<K, V, B> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V, B>;
    fn into_iter(self) -> Iter<'a, K, V, B> {
        self.iter()
    }
}
impl<'a, K, V, const B: usize> IntoIterator for &'a mut BTreeMap<K, V, B> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V, B>;
    fn into_iter(self) -> IterMut<'a, K, V, B> {
        self.iter_mut()
    }
}
impl<K, V, const B: usize> Clone for BTreeMap<K, V, B>
where
    K: Clone + Ord,
    V: Clone,
{
    fn clone(&self) -> BTreeMap<K, V, B> {
        let mut map = BTreeMap::new();
        for (k, v) in self.iter() {
            map.insert(k.clone(), v.clone());
        }
        map
    }
}
impl<K: Ord, V, const B: usize> FromIterator<(K, V)> for BTreeMap<K, V, B> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> BTreeMap<K, V, B> {
        let mut map = BTreeMap::new();
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}
impl<K, V, const B: usize, const N: usize> From<[(K, V); N]> for BTreeMap<K, V, B>
where
    K: Ord,
{
    fn from(arr: [(K, V); N]) -> BTreeMap<K, V, B> {
        let mut map = BTreeMap::new();
        for (k, v) in arr {
            map.insert(k, v);
        }
        map
    }
}
impl<K, V, const B: usize> Extend<(K, V)> for BTreeMap<K, V, B>
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
impl<'a, K, V, const B: usize> Extend<(&'a K, &'a V)> for BTreeMap<K, V, B>
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
impl<K, Q, V, const B: usize> std::ops::Index<&Q> for BTreeMap<K, V, B>
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
impl<K: Debug, V: Debug, const B: usize> Debug for BTreeMap<K, V, B> {
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
impl<K, V, const B: usize> Serialize for BTreeMap<K, V, B>
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
struct BTreeMapVisitor<K, V, const B: usize> {
    marker: PhantomData<fn() -> BTreeMap<K, V, B>>,
}

#[cfg(feature = "serde")]
impl<K, V, const B: usize> BTreeMapVisitor<K, V, B> {
    fn new() -> Self {
        BTreeMapVisitor {
            marker: PhantomData,
        }
    }
}

#[cfg(feature = "serde")]
impl<'de, K, V, const B: usize> Visitor<'de> for BTreeMapVisitor<K, V, B>
where
    K: Deserialize<'de> + Ord,
    V: Deserialize<'de>,
{
    // The type that our Visitor is going to produce.
    type Value = BTreeMap<K, V, B>;

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
impl<'de, K, V, const B: usize> Deserialize<'de> for BTreeMap<K, V, B>
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

enum Tree<K, V, const B: usize> {
    L(Leaf<K, V, B>),
    NL(NonLeaf<K, V, B>),
}
impl<K, V, const B: usize> Default for Tree<K, V, B> {
    fn default() -> Self {
        Tree::L(Leaf(LeafVec::new()))
    }
}
impl<K, V, const B: usize> Tree<K, V, B> {
    fn insert(&mut self, key: K, x: &mut InsertCtx<K, V, B>)
    where
        K: Ord,
    {
        match self {
            Tree::L(leaf) => leaf.insert(key, x),
            Tree::NL(nonleaf) => nonleaf.insert(key, x),
        }
    }

    fn prepare_insert(&mut self, pos: &mut PosVec, level: usize) -> Option<Split<K, V, B>> {
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

    fn new_root(&mut self, (med, right): Split<K, V, B>) {
        let left = std::mem::take(self);
        let mut v = FixedCapVec::new();
        v.push(med);
        let mut c = FixedCapVec::new();
        c.push(left);
        c.push(right);
        *self = Tree::NL(NonLeaf { v, c });
    }

    fn nonleaf(&mut self) -> &mut NonLeaf<K, V, B> {
        match self {
            Tree::NL(nl) => nl,
            _ => panic!(),
        }
    }

    fn leaf(&mut self) -> &mut Leaf<K, V, B> {
        match self {
            Tree::L(leaf) => leaf,
            _ => panic!(),
        }
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

    fn find_position<Q>(&mut self, key: &Q, pos: &mut Position<K, V, B>)
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

    fn iter_mut(&mut self) -> RangeMut<'_, K, V, B> {
        let mut x = RangeMut::new();
        x.push_tree(self, true);
        x
    }

    fn iter(&self) -> Range<'_, K, V, B> {
        let mut x = Range::new();
        x.push_tree(self, true);
        x
    }

    fn range_mut<T, R>(&mut self, range: &R) -> RangeMut<'_, K, V, B>
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        let mut x = RangeMut::new();
        x.push_range(self, range, true);
        x
    }

    fn range<T, R>(&self, range: &R) -> Range<'_, K, V, B>
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        let mut x = Range::new();
        x.push_range(self, range, true);
        x
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

struct InsertCtx<K, V, const B: usize> {
    value: Option<V>,
    split: Option<Split<K, V, B>>,
}

struct Leaf<K, V, const B: usize>(LeafVec<K, V, B>);
impl<K, V, const B: usize> Leaf<K, V, B> {
    fn full(&self) -> bool {
        self.0.len() == B
    }

    fn get_lower<Q>(&self, bound: Bound<&Q>) -> usize
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match bound {
            Bound::Unbounded => 0,
            Bound::Included(k) => match self.0.search(|kv| kv.0.borrow().cmp(k)) {
                Ok(x) => x,
                Err(x) => x,
            },
            Bound::Excluded(k) => match self.0.search(|kv| kv.0.borrow().cmp(k)) {
                Ok(x) => x + 1,
                Err(x) => x,
            },
        }
    }

    fn get_upper<Q>(&self, bound: Bound<&Q>) -> usize
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match bound {
            Bound::Unbounded => self.0.len(),
            Bound::Included(k) => match self.0.search(|x| x.0.borrow().cmp(k)) {
                Ok(x) => x + 1,
                Err(x) => x,
            },
            Bound::Excluded(k) => match self.0.search(|x| x.0.borrow().cmp(k)) {
                Ok(x) => x,
                Err(x) => x,
            },
        }
    }

    fn split(&mut self) -> ((K, V), LeafVec<K, V, B>) {
        let right = self.0.split_off(B / 2 + 1);
        let med = self.0.pop().unwrap();
        (med, right)
    }

    fn insert(&mut self, key: K, x: &mut InsertCtx<K, V, B>)
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
            if i > B / 2 {
                i -= B / 2 + 1;
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

    fn prepare_insert(&mut self, pos: &mut PosVec) -> Option<Split<K, V, B>> {
        debug_assert!(self.full());
        let mut level = pos.len() - 1;
        if level == 0 {
            level += 1;
            pos.insert(0, 0);
        }
        if pos[level] > (B / 2) as u8 {
            pos[level] -= (B / 2 + 1) as u8;
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

    fn find_position<Q>(&mut self, key: &Q, pos: &mut Position<K, V, B>)
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

    fn get_xy<T, R>(&self, range: &R) -> (usize, usize)
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        // ToDo : use some kind of binary search.
        let mut x = 0;
        while x < self.0.len() && !range.contains(self.0.ix(x).0.borrow()) {
            x += 1;
        }
        let mut y = self.0.len();
        while y > x && !range.contains(self.0.ix(y - 1).0.borrow()) {
            y -= 1;
        }
        (x, y)
    }

    fn iter_mut(&mut self) -> IterLeafMut<'_, K, V> {
        IterLeafMut(self.0.iter_mut())
    }

    fn iter(&self) -> IterLeaf<'_, K, V> {
        IterLeaf(self.0.iter())
    }
} // End impl Leaf

struct NonLeaf<K, V, const B: usize> {
    v: NonLeafVec<K, V, B>,
    c: NonLeafChildVec<K, V, B>,
}
impl<K, V, const B: usize> NonLeaf<K, V, B> {
    fn full(&self) -> bool {
        self.v.len() == B - 1
    }

    fn get_lower<Q>(&self, bound: Bound<&Q>) -> usize
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match bound {
            Bound::Unbounded => 0,
            Bound::Included(k) => match self.v.search(|kv| kv.0.borrow().cmp(k)) {
                Ok(x) => x,
                Err(x) => x,
            },
            Bound::Excluded(k) => match self.v.search(|kv| kv.0.borrow().cmp(k)) {
                Ok(x) => x + 1,
                Err(x) => x,
            },
        }
    }

    fn get_upper<Q>(&self, bound: Bound<&Q>) -> usize
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match bound {
            Bound::Unbounded => self.v.len(),
            Bound::Included(k) => match self.v.search(|kv| kv.0.borrow().cmp(k)) {
                Ok(x) => x + 1,
                Err(x) => x,
            },
            Bound::Excluded(k) => match self.v.search(|kv| kv.0.borrow().cmp(k)) {
                Ok(x) => x,
                Err(x) => x,
            },
        }
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

    fn find_position<Q>(&mut self, key: &Q, pos: &mut Position<K, V, B>)
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

    fn split(&mut self) -> Split<K, V, B> {
        let right = Self {
            v: self.v.split_off(B / 2 + 1),
            c: self.c.split_off(B / 2 + 1),
        };
        let med = self.v.pop().unwrap();
        (med, Tree::NL(right))
    }

    fn insert(&mut self, key: K, x: &mut InsertCtx<K, V, B>)
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

    fn prepare_insert(&mut self, pos: &mut PosVec, mut level: usize) -> Option<Split<K, V, B>> {
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
            if pos[level] > (B / 2) as u8 {
                pos[level] -= (B / 2 + 1) as u8;
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
} // End impl NonLeaf

/// Error returned by [BTreeMap::try_insert].
pub struct OccupiedError<'a, K, V, const B: usize>
where
    K: 'a,
    V: 'a,
{
    /// Occupied entry, has the key that was not inserted.
    pub entry: OccupiedEntry<'a, K, V, B>,
    /// Value that was not inserted.
    pub value: V,
}

/// Entry in BTreeMap, returned by [BTreeMap::entry].
pub enum Entry<'a, K, V, const B: usize> {
    /// Vacant entry - map doesn't yet contain key.
    Vacant(VacantEntry<'a, K, V, B>),
    /// Occupied entry - map already contains key.
    Occupied(OccupiedEntry<'a, K, V, B>),
}
impl<'a, K, V, const B: usize> Entry<'a, K, V, B>
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
    pub fn and_modify<F>(mut self, f: F) -> Entry<'a, K, V, B>
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

enum TreePtr<K, V, const B: usize> {
    None,
    L(*mut Leaf<K, V, B>, usize),
    NL(*mut NonLeaf<K, V, B>, usize),
}
unsafe impl<K: Send, V: Send, const B: usize> Send for TreePtr<K, V, B> {}
unsafe impl<K: Sync, V: Send, const B: usize> Sync for TreePtr<K, V, B> {}
impl<K, V, const B: usize> TreePtr<K, V, B> {
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
struct Position<K, V, const B: usize> {
    key_found: bool,
    ix: PosVec,
    ptr: TreePtr<K, V, B>,
}
impl<K, V, const B: usize> Position<K, V, B> {
    fn new() -> Self {
        Self {
            key_found: false,
            ix: PosVec::new(),
            ptr: TreePtr::None,
        }
    }
}

/// Vacant [Entry].
pub struct VacantEntry<'a, K, V, const B: usize> {
    map: *mut BTreeMap<K, V, B>,
    key: K,
    pos: Position<K, V, B>,
    _pd: PhantomData<&'a mut BTreeMap<K, V, B>>,
}
unsafe impl<'a, K: Send, V: Send, const B: usize> Send for VacantEntry<'a, K, V, B> {}
unsafe impl<'a, K: Sync, V: Send, const B: usize> Sync for VacantEntry<'a, K, V, B> {}

impl<'a, K, V, const B: usize> VacantEntry<'a, K, V, B>
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
        match self.pos.ptr {
            TreePtr::L(ptr, ix) => unsafe {
                let x = &mut (*ptr).0;
                x.insert(ix, (self.key, value));
                let result = &mut x.ixm(ix).1;
                (*self.map).len += 1;
                result
            },
            _ => unsafe { &mut (*self.map).ins_pos(&mut self.pos, self.key, value).1 },
        }
    }
}

enum OccupiedEntryKey<K, V, const B: usize> {
    First,
    Last,
    Some(Position<K, V, B>),
}

/// Occupied [Entry].
pub struct OccupiedEntry<'a, K, V, const B: usize> {
    map: *mut BTreeMap<K, V, B>,
    key: OccupiedEntryKey<K, V, B>,
    _pd: PhantomData<&'a mut BTreeMap<K, V, B>>,
}
unsafe impl<'a, K: Send, V: Send, const B: usize> Send for OccupiedEntry<'a, K, V, B> {}
unsafe impl<'a, K: Sync, V: Send, const B: usize> Sync for OccupiedEntry<'a, K, V, B> {}

impl<'a, K, V, const B: usize> OccupiedEntry<'a, K, V, B>
where
    K: Ord,
{
    /// Get reference to entry key.
    pub fn key(&self) -> &K {
        unsafe {
            match &self.key {
                OccupiedEntryKey::Some(pos) => pos.ptr.key_ref(),
                OccupiedEntryKey::First => (*self.map).first_key_value().unwrap().0,
                OccupiedEntryKey::Last => (*self.map).last_key_value().unwrap().0,
            }
        }
    }

    /// Remove (key,value) from map, returning key and value.
    pub fn remove_entry(self) -> (K, V) {
        unsafe {
            match &self.key {
                OccupiedEntryKey::Some(pos) => {
                    let result = match pos.ptr {
                        TreePtr::L(ptr, ix) => (*ptr).0.remove(ix),
                        TreePtr::NL(ptr, ix) => (*ptr).remove_at(ix),
                        TreePtr::None => panic!(),
                    };
                    (*self.map).len -= 1;
                    result
                }
                OccupiedEntryKey::First => (*self.map).pop_first().unwrap(),
                OccupiedEntryKey::Last => (*self.map).pop_last().unwrap(),
            }
        }
    }

    /// Remove (key,value) from map, returning the value.
    pub fn remove(self) -> V {
        self.remove_entry().1
    }

    /// Get reference to the value.
    pub fn get(&self) -> &V {
        unsafe {
            match &self.key {
                OccupiedEntryKey::Some(pos) => pos.ptr.value_ref(),
                OccupiedEntryKey::First => (*self.map).first_key_value().unwrap().1,
                OccupiedEntryKey::Last => (*self.map).last_key_value().unwrap().1,
            }
        }
    }

    /// Get mutable reference to the value.
    pub fn get_mut(&mut self) -> &mut V {
        unsafe {
            match &mut self.key {
                OccupiedEntryKey::Some(pos) => pos.ptr.value_mut(),
                OccupiedEntryKey::First => (*self.map).first_key_value_mut().unwrap().1,
                OccupiedEntryKey::Last => (*self.map).last_key_value_mut().unwrap().1,
            }
        }
    }

    /// Get mutable reference to the value, consuming the entry.
    pub fn into_mut(mut self) -> &'a mut V {
        unsafe {
            match &mut self.key {
                OccupiedEntryKey::Some(pos) => match pos.ptr {
                    TreePtr::None => panic!(),
                    TreePtr::L(ptr, ix) => &mut (*ptr).0.ixm(ix).1,
                    TreePtr::NL(ptr, ix) => &mut (*ptr).v.ixm(ix).1,
                },
                OccupiedEntryKey::First => (*self.map).first_key_value_mut().unwrap().1,
                OccupiedEntryKey::Last => (*self.map).last_key_value_mut().unwrap().1,
            }
        }
    }

    /// Update the value returns the old value.
    pub fn insert(&mut self, value: V) -> V {
        std::mem::replace(self.get_mut(), value)
    }
}

// Mutable reference iteration.

struct StkMut<'a, K, V, const B: usize> {
    v: std::slice::IterMut<'a, (K, V)>,
    c: std::slice::IterMut<'a, Tree<K, V, B>>,
}

enum StealResultMut<'a, K, V, const B: usize> {
    KV((&'a K, &'a mut V)),    // Key-value pair.
    CT(&'a mut Tree<K, V, B>), // Child Tree.
    Nothing,
}

/// Iterator returned by [BTreeMap::iter_mut].
pub struct IterMut<'a, K, V, const B: usize> {
    len: usize,
    inner: RangeMut<'a, K, V, B>,
}
impl<'a, K, V, const B: usize> Iterator for IterMut<'a, K, V, B> {
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
impl<'a, K, V, const B: usize> ExactSizeIterator for IterMut<'a, K, V, B> {
    fn len(&self) -> usize {
        self.len
    }
}
impl<'a, K, V, const B: usize> DoubleEndedIterator for IterMut<'a, K, V, B> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            self.inner.next_back()
        }
    }
}
impl<'a, K, V, const B: usize> FusedIterator for IterMut<'a, K, V, B> {}

/// Iterator returned by [BTreeMap::range_mut].
pub struct RangeMut<'a, K, V, const B: usize> {
    /* There are two iterations going on to implement DoubleEndedIterator.
       fwd_leaf and fwd_stk are initially used for forward (next) iteration,
       once they are exhausted, key-value pairs and child trees are "stolen" from
       bck_stk and bck_leaf which are (conversely) initially used for next_back iteration.
    */
    fwd_leaf: Option<IterLeafMut<'a, K, V>>,
    bck_leaf: Option<IterLeafMut<'a, K, V>>,
    fwd_stk: StkMutVec<'a, K, V, B>,
    bck_stk: StkMutVec<'a, K, V, B>,
}
impl<'a, K, V, const B: usize> RangeMut<'a, K, V, B> {
    fn new() -> Self {
        Self {
            fwd_leaf: None,
            bck_leaf: None,
            fwd_stk: StkMutVec::new(),
            bck_stk: StkMutVec::new(),
        }
    }
    fn push_tree(&mut self, tree: &'a mut Tree<K, V, B>, both: bool) {
        match tree {
            Tree::L(x) => {
                self.fwd_leaf = Some(x.iter_mut());
            }
            Tree::NL(x) => {
                let (v, mut c) = (x.v.iter_mut(), x.c.iter_mut());
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
    fn push_range<T, R>(&mut self, tree: &'a mut Tree<K, V, B>, range: &R, both: bool)
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        match tree {
            Tree::L(leaf) => {
                let (x, y) = leaf.get_xy(range);
                self.fwd_leaf = Some(IterLeafMut(leaf.0[x..y].iter_mut()));
            }
            Tree::NL(t) => {
                let (x, y) = t.get_xy(range);
                let (v, mut c) = (t.v[x..y].iter_mut(), t.c[x..y + 1].iter_mut());

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
    fn push_range_back<T, R>(&mut self, tree: &'a mut Tree<K, V, B>, range: &R)
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        match tree {
            Tree::L(leaf) => {
                let (x, y) = leaf.get_xy(range);
                self.bck_leaf = Some(IterLeafMut(leaf.0[x..y].iter_mut()));
            }
            Tree::NL(t) => {
                let (x, y) = t.get_xy(range);
                let (v, mut c) = (t.v[x..y].iter_mut(), t.c[x..y + 1].iter_mut());

                let ct_back = c.next_back();

                self.bck_stk.push(StkMut { v, c });
                if let Some(ct_back) = ct_back {
                    self.push_range_back(ct_back, range);
                }
            }
        }
    }
    fn push_tree_back(&mut self, tree: &'a mut Tree<K, V, B>) {
        match tree {
            Tree::L(x) => {
                self.bck_leaf = Some(x.iter_mut());
            }
            Tree::NL(x) => {
                let (v, mut c) = (x.v.iter_mut(), x.c.iter_mut());
                let ct_back = c.next_back();
                self.bck_stk.push(StkMut { v, c });
                if let Some(ct_back) = ct_back {
                    self.push_tree_back(ct_back);
                }
            }
        }
    }
    fn steal_bck(&mut self) -> StealResultMut<'a, K, V, B> {
        for s in self.bck_stk.iter_mut() {
            if s.v.len() > s.c.len() {
                let kv = s.v.next().unwrap();
                return StealResultMut::KV((&kv.0, &mut kv.1));
            } else if let Some(ct) = s.c.next() {
                return StealResultMut::CT(ct);
            }
        }
        StealResultMut::Nothing
    }
    fn steal_fwd(&mut self) -> StealResultMut<'a, K, V, B> {
        for s in self.fwd_stk.iter_mut() {
            if s.v.len() > s.c.len() {
                let kv = s.v.next_back().unwrap();
                return StealResultMut::KV((&kv.0, &mut kv.1));
            } else if let Some(ct) = s.c.next_back() {
                return StealResultMut::CT(ct);
            }
        }
        StealResultMut::Nothing
    }
}
impl<'a, K, V, const B: usize> Iterator for RangeMut<'a, K, V, B> {
    type Item = (&'a K, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(f) = &mut self.fwd_leaf {
                if let Some(x) = f.next() {
                    return Some(x);
                } else {
                    self.fwd_leaf = None;
                }
            } else if let Some(s) = self.fwd_stk.last_mut() {
                if let Some(kv) = s.v.next() {
                    if let Some(ct) = s.c.next() {
                        self.push_tree(ct, false);
                    }
                    return Some((&kv.0, &mut kv.1));
                } else {
                    self.fwd_stk.pop();
                }
            } else {
                match self.steal_bck() {
                    StealResultMut::KV(kv) => {
                        return Some(kv);
                    }
                    StealResultMut::CT(ct) => {
                        self.push_tree(ct, false);
                    }
                    StealResultMut::Nothing => {
                        if let Some(f) = &mut self.bck_leaf {
                            if let Some(x) = f.next() {
                                return Some(x);
                            } else {
                                self.bck_leaf = None;
                                return None;
                            }
                        } else {
                            return None;
                        }
                    }
                }
            }
        }
    }
}
impl<'a, K, V, const B: usize> DoubleEndedIterator for RangeMut<'a, K, V, B> {
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(f) = &mut self.bck_leaf {
                if let Some(x) = f.next_back() {
                    return Some(x);
                } else {
                    self.bck_leaf = None;
                }
            } else if let Some(s) = self.bck_stk.last_mut() {
                if let Some(kv) = s.v.next_back() {
                    if let Some(ct) = s.c.next_back() {
                        self.push_tree_back(ct);
                    }
                    return Some((&kv.0, &mut kv.1));
                } else {
                    self.bck_stk.pop();
                }
            } else {
                match self.steal_fwd() {
                    StealResultMut::KV(kv) => {
                        return Some(kv);
                    }
                    StealResultMut::CT(ct) => {
                        self.push_tree_back(ct);
                    }
                    StealResultMut::Nothing => {
                        if let Some(f) = &mut self.fwd_leaf {
                            if let Some(x) = f.next_back() {
                                return Some(x);
                            } else {
                                self.fwd_leaf = None;
                                return None;
                            }
                        } else {
                            return None;
                        }
                    }
                }
            }
        }
    }
}
impl<'a, K, V, const B: usize> FusedIterator for RangeMut<'a, K, V, B> {}

// Consuming iteration.

struct StkCon<K, V, const B: usize> {
    v: FixedCapIter<(K, V), B>,
    c: FixedCapIter<Tree<K, V, B>, B>,
}

enum StealResultCon<K, V, const B: usize> {
    KV((K, V)),        // Key-value pair.
    CT(Tree<K, V, B>), // Child Tree.
    Nothing,
}

/// Consuming iterator returned by [BTreeMap::into_iter].
pub struct IntoIter<K, V, const B: usize> {
    len: usize,
    inner: IntoIterInner<K, V, B>,
}
impl<K, V, const B: usize> IntoIter<K, V, B> {
    fn new(bt: BTreeMap<K, V, B>) -> Self {
        let mut s = Self {
            len: bt.len(),
            inner: IntoIterInner::new(),
        };
        s.inner.push_tree(bt.tree, true);
        s
    }
}

impl<K, V, const B: usize> Iterator for IntoIter<K, V, B> {
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        let result = self.inner.next();
        if result.is_some() {
            self.len -= 1;
        }
        result
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}
impl<K, V, const B: usize> DoubleEndedIterator for IntoIter<K, V, B> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let result = self.inner.next_back();
        if result.is_some() {
            self.len -= 1;
        }
        result
    }
}
impl<K, V, const B: usize> FusedIterator for IntoIter<K, V, B> {}

struct IntoIterInner<K, V, const B: usize> {
    fwd_leaf: Option<FixedCapIter<(K, V), B>>,
    bck_leaf: Option<FixedCapIter<(K, V), B>>,
    fwd_stk: StkConVec<K, V, B>,
    bck_stk: StkConVec<K, V, B>,
}
impl<K, V, const B: usize> IntoIterInner<K, V, B> {
    fn new() -> Self {
        Self {
            fwd_leaf: None,
            bck_leaf: None,
            fwd_stk: StkConVec::new(),
            bck_stk: StkConVec::new(),
        }
    }
    fn push_tree(&mut self, tree: Tree<K, V, B>, both: bool) {
        match tree {
            Tree::L(x) => {
                self.fwd_leaf = Some(x.0.into_iter());
            }
            Tree::NL(x) => {
                let (v, mut c) = (x.v.into_iter(), x.c.into_iter());
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
    fn push_tree_back(&mut self, tree: Tree<K, V, B>) {
        match tree {
            Tree::L(x) => {
                self.bck_leaf = Some(x.0.into_iter());
            }
            Tree::NL(x) => {
                let (v, mut c) = (x.v.into_iter(), x.c.into_iter());
                let ct_back = c.next_back();
                self.bck_stk.push(StkCon { v, c });
                if let Some(ct_back) = ct_back {
                    self.push_tree_back(ct_back);
                }
            }
        }
    }
    fn steal_bck(&mut self) -> StealResultCon<K, V, B> {
        for s in self.bck_stk.iter_mut() {
            if s.v.len() > s.c.len() {
                let kv = s.v.next().unwrap();
                return StealResultCon::KV(kv);
            } else if let Some(ct) = s.c.next() {
                return StealResultCon::CT(ct);
            }
        }
        StealResultCon::Nothing
    }
    fn steal_fwd(&mut self) -> StealResultCon<K, V, B> {
        for s in self.fwd_stk.iter_mut() {
            if s.v.len() > s.c.len() {
                let kv = s.v.next_back().unwrap();
                return StealResultCon::KV(kv);
            } else if let Some(ct) = s.c.next_back() {
                return StealResultCon::CT(ct);
            }
        }
        StealResultCon::Nothing
    }
}
impl<K, V, const B: usize> Iterator for IntoIterInner<K, V, B> {
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(f) = &mut self.fwd_leaf {
                if let Some(x) = f.next() {
                    return Some(x);
                } else {
                    self.fwd_leaf = None;
                }
            } else if let Some(s) = self.fwd_stk.last_mut() {
                if let Some(kv) = s.v.next() {
                    if let Some(ct) = s.c.next() {
                        self.push_tree(ct, false);
                    }
                    return Some(kv);
                } else {
                    self.fwd_stk.pop();
                }
            } else {
                match self.steal_bck() {
                    StealResultCon::KV(kv) => {
                        return Some(kv);
                    }
                    StealResultCon::CT(ct) => {
                        self.push_tree(ct, false);
                    }
                    StealResultCon::Nothing => {
                        if let Some(f) = &mut self.bck_leaf {
                            if let Some(x) = f.next() {
                                return Some(x);
                            } else {
                                self.bck_leaf = None;
                                return None;
                            }
                        } else {
                            return None;
                        }
                    }
                }
            }
        }
    }
}
impl<K, V, const B: usize> DoubleEndedIterator for IntoIterInner<K, V, B> {
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(f) = &mut self.bck_leaf {
                if let Some(x) = f.next_back() {
                    return Some(x);
                } else {
                    self.bck_leaf = None;
                }
            } else if let Some(s) = self.bck_stk.last_mut() {
                if let Some(kv) = s.v.next_back() {
                    if let Some(ct) = s.c.next_back() {
                        self.push_tree_back(ct);
                    }
                    return Some(kv);
                } else {
                    self.bck_stk.pop();
                }
            } else {
                match self.steal_fwd() {
                    StealResultCon::KV(kv) => {
                        return Some(kv);
                    }
                    StealResultCon::CT(ct) => {
                        self.push_tree_back(ct);
                    }
                    StealResultCon::Nothing => {
                        if let Some(f) = &mut self.fwd_leaf {
                            if let Some(x) = f.next_back() {
                                return Some(x);
                            } else {
                                self.fwd_leaf = None;
                                return None;
                            }
                        } else {
                            return None;
                        }
                    }
                }
            }
        }
    }
}

// Immutable reference iteration.

struct Stk<'a, K, V, const B: usize> {
    v: std::slice::Iter<'a, (K, V)>,
    c: std::slice::Iter<'a, Tree<K, V, B>>,
}

enum StealResult<'a, K, V, const B: usize> {
    KV((&'a K, &'a V)),    // Key-value pair.
    CT(&'a Tree<K, V, B>), // Child Tree.
    Nothing,
}

/// Iterator returned by [BTreeMap::iter].
pub struct Iter<'a, K, V, const B: usize> {
    len: usize,
    inner: Range<'a, K, V, B>,
}
impl<'a, K, V, const B: usize> Iterator for Iter<'a, K, V, B> {
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
impl<'a, K, V, const B: usize> ExactSizeIterator for Iter<'a, K, V, B> {
    fn len(&self) -> usize {
        self.len
    }
}
impl<'a, K, V, const B: usize> DoubleEndedIterator for Iter<'a, K, V, B> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            self.inner.next_back()
        }
    }
}
impl<'a, K, V, const B: usize> FusedIterator for Iter<'a, K, V, B> {}

/// Iterator returned by [BTreeMap::range].
pub struct Range<'a, K, V, const B: usize> {
    fwd_leaf: Option<IterLeaf<'a, K, V>>,
    bck_leaf: Option<IterLeaf<'a, K, V>>,
    fwd_stk: StkVec<'a, K, V, B>,
    bck_stk: StkVec<'a, K, V, B>,
}
impl<'a, K, V, const B: usize> Range<'a, K, V, B> {
    fn new() -> Self {
        Self {
            fwd_leaf: None,
            bck_leaf: None,
            fwd_stk: StkVec::new(),
            bck_stk: StkVec::new(),
        }
    }
    fn push_tree(&mut self, tree: &'a Tree<K, V, B>, both: bool) {
        match tree {
            Tree::L(x) => {
                self.fwd_leaf = Some(x.iter());
            }
            Tree::NL(x) => {
                let (v, mut c) = (x.v.iter(), x.c.iter());
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
    fn push_range<T, R>(&mut self, tree: &'a Tree<K, V, B>, range: &R, both: bool)
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        match tree {
            Tree::L(leaf) => {
                let (x, y) = leaf.get_xy(range);
                self.fwd_leaf = Some(IterLeaf(leaf.0[x..y].iter()));
            }
            Tree::NL(t) => {
                let (x, y) = t.get_xy(range);
                let (v, mut c) = (t.v[x..y].iter(), t.c[x..y + 1].iter());

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
    fn push_range_back<T, R>(&mut self, tree: &'a Tree<K, V, B>, range: &R)
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        match tree {
            Tree::L(leaf) => {
                let (x, y) = leaf.get_xy(range);
                self.bck_leaf = Some(IterLeaf(leaf.0[x..y].iter()));
            }
            Tree::NL(t) => {
                let (x, y) = t.get_xy(range);
                let (v, mut c) = (t.v[x..y].iter(), t.c[x..y + 1].iter());
                let ct_back = c.next_back();
                self.bck_stk.push(Stk { v, c });
                if let Some(ct_back) = ct_back {
                    self.push_range_back(ct_back, range);
                }
            }
        }
    }
    fn push_tree_back(&mut self, tree: &'a Tree<K, V, B>) {
        match tree {
            Tree::L(x) => {
                self.bck_leaf = Some(x.iter());
            }
            Tree::NL(x) => {
                let (v, mut c) = (x.v.iter(), x.c.iter());
                let ct_back = c.next_back();
                self.bck_stk.push(Stk { v, c });
                if let Some(ct_back) = ct_back {
                    self.push_tree_back(ct_back);
                }
            }
        }
    }
    fn steal_bck(&mut self) -> StealResult<'a, K, V, B> {
        for s in self.bck_stk.iter_mut() {
            if s.v.len() > s.c.len() {
                let kv = s.v.next().unwrap();
                return StealResult::KV((&kv.0, &kv.1));
            } else if let Some(ct) = s.c.next() {
                return StealResult::CT(ct);
            }
        }
        StealResult::Nothing
    }
    fn steal_fwd(&mut self) -> StealResult<'a, K, V, B> {
        for s in self.fwd_stk.iter_mut() {
            if s.v.len() > s.c.len() {
                let kv = s.v.next_back().unwrap();
                return StealResult::KV((&kv.0, &kv.1));
            } else if let Some(ct) = s.c.next_back() {
                return StealResult::CT(ct);
            }
        }
        StealResult::Nothing
    }
}
impl<'a, K, V, const B: usize> Iterator for Range<'a, K, V, B> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(f) = &mut self.fwd_leaf {
                if let Some(x) = f.next() {
                    return Some(x);
                } else {
                    self.fwd_leaf = None;
                }
            } else if let Some(s) = self.fwd_stk.last_mut() {
                if let Some(kv) = s.v.next() {
                    if let Some(ct) = s.c.next() {
                        self.push_tree(ct, false);
                    }
                    return Some((&kv.0, &kv.1));
                } else {
                    self.fwd_stk.pop();
                }
            } else {
                match self.steal_bck() {
                    StealResult::KV(kv) => {
                        return Some(kv);
                    }
                    StealResult::CT(ct) => {
                        self.push_tree(ct, false);
                    }
                    StealResult::Nothing => {
                        if let Some(f) = &mut self.bck_leaf {
                            if let Some(x) = f.next() {
                                return Some(x);
                            } else {
                                self.bck_leaf = None;
                                return None;
                            }
                        } else {
                            return None;
                        }
                    }
                }
            }
        }
    }
}
impl<'a, K, V, const B: usize> DoubleEndedIterator for Range<'a, K, V, B> {
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(f) = &mut self.bck_leaf {
                if let Some(x) = f.next_back() {
                    return Some(x);
                } else {
                    self.bck_leaf = None;
                }
            } else if let Some(s) = self.bck_stk.last_mut() {
                if let Some(kv) = s.v.next_back() {
                    if let Some(ct) = s.c.next_back() {
                        self.push_tree_back(ct);
                    }
                    return Some((&kv.0, &kv.1));
                } else {
                    self.bck_stk.pop();
                }
            } else {
                match self.steal_fwd() {
                    StealResult::KV(kv) => {
                        return Some(kv);
                    }
                    StealResult::CT(ct) => {
                        self.push_tree_back(ct);
                    }
                    StealResult::Nothing => {
                        if let Some(f) = &mut self.fwd_leaf {
                            if let Some(x) = f.next_back() {
                                return Some(x);
                            } else {
                                self.fwd_leaf = None;
                                return None;
                            }
                        } else {
                            return None;
                        }
                    }
                }
            }
        }
    }
}
impl<'a, K, V, const B: usize> FusedIterator for Range<'a, K, V, B> {}

/// Consuming iterator returned by [BTreeMap::into_keys].
pub struct IntoKeys<K, V, const B: usize>(IntoIter<K, V, B>);
impl<K, V, const B: usize> Iterator for IntoKeys<K, V, B> {
    type Item = K;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.0.next()?.0)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<K, V, const B: usize> DoubleEndedIterator for IntoKeys<K, V, B> {
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(self.0.next_back()?.0)
    }
}
impl<K, V, const B: usize> FusedIterator for IntoKeys<K, V, B> {}

/// Consuming iterator returned by [BTreeMap::into_values].
pub struct IntoValues<K, V, const B: usize>(IntoIter<K, V, B>);
impl<K, V, const B: usize> Iterator for IntoValues<K, V, B> {
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.0.next()?.1)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<K, V, const B: usize> DoubleEndedIterator for IntoValues<K, V, B> {
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(self.0.next_back()?.1)
    }
}
impl<K, V, const B: usize> FusedIterator for IntoValues<K, V, B> {}

// Leaf iterators.

struct IterLeafMut<'a, K, V>(std::slice::IterMut<'a, (K, V)>);
impl<'a, K, V> Iterator for IterLeafMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);
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

// Trivial iterators.

/// Iterator returned by [BTreeMap::values_mut].
pub struct ValuesMut<'a, K, V, const B: usize>(IterMut<'a, K, V, B>);
impl<'a, K, V, const B: usize> Iterator for ValuesMut<'a, K, V, B> {
    type Item = &'a mut V;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(_, v)| v)
    }
}
impl<'a, K, V, const B: usize> DoubleEndedIterator for ValuesMut<'a, K, V, B> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|(_, v)| v)
    }
}
impl<'a, K, V, const B: usize> FusedIterator for ValuesMut<'a, K, V, B> {}

/// Iterator returned by [BTreeMap::values].
pub struct Values<'a, K, V, const B: usize>(Iter<'a, K, V, B>);
impl<'a, K, V, const B: usize> Iterator for Values<'a, K, V, B> {
    type Item = &'a V;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(_, v)| v)
    }
}
impl<'a, K, V, const B: usize> DoubleEndedIterator for Values<'a, K, V, B> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|(_, v)| v)
    }
}
impl<'a, K, V, const B: usize> FusedIterator for Values<'a, K, V, B> {}

/// Iterator returned by [BTreeMap::keys].
pub struct Keys<'a, K, V, const B: usize>(Iter<'a, K, V, B>);
impl<'a, K, V, const B: usize> Iterator for Keys<'a, K, V, B> {
    type Item = &'a K;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(k, _)| k)
    }
}
impl<'a, K, V, const B: usize> DoubleEndedIterator for Keys<'a, K, V, B> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|(k, _)| k)
    }
}
impl<'a, K, V, const B: usize> FusedIterator for Keys<'a, K, V, B> {}

/// Iterator returned by [BTreeMap::extract_if].
// #[derive(Debug)]
pub struct ExtractIf<'a, K, V, const B: usize, F>
where
    F: FnMut(&K, &mut V) -> bool,
{
    source: CursorMut<'a, K, V, B>,
    pred: F,
}
impl<K, V, const B: usize, F> fmt::Debug for ExtractIf<'_, K, V, B, F>
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
impl<'a, K, V, const B: usize, F> Iterator for ExtractIf<'a, K, V, B, F>
where
    F: FnMut(&K, &mut V) -> bool,
{
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (k, v) = self.source.peek_next()?;
            if (self.pred)(k, v) {
                return self.source.remove_next();
            } else {
                self.source.next();
            }
        }
    }
}
impl<'a, K, V, const B: usize, F> FusedIterator for ExtractIf<'a, K, V, B, F> where
    F: FnMut(&K, &mut V) -> bool
{
}

// Cursors.

/// Error type for [CursorMut::insert_before] and [CursorMut::insert_after].
#[derive(Debug, Clone)]
pub struct UnorderedKeyError {}

/// Cursor that allows mutation of map, returned by [BTreeMap::lower_bound_mut], [BTreeMap::upper_bound_mut].
pub struct CursorMut<'a, K, V, const B: usize>(CursorMutKey<'a, K, V, B>);
impl<'a, K, V, const B: usize> CursorMut<'a, K, V, B> {
    fn lower_bound<Q>(map: &mut BTreeMap<K, V, B>, bound: Bound<&Q>) -> Self
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        unsafe {
            // Converting map to raw pointer here is necessary to keep Miri happy
            // although not when using MIRIFLAGS=-Zmiri-tree-borrows.
            let map: *mut BTreeMap<K, V, B> = map;
            let mut s = CursorMutKey::make(map);
            s.push_lower(&mut (*map).tree, bound);
            Self(s)
        }
    }

    fn upper_bound<Q>(map: &mut BTreeMap<K, V, B>, bound: Bound<&Q>) -> Self
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        unsafe {
            let map: *mut BTreeMap<K, V, B> = map;
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

    /// Returns references to the previous key/value pair.
    pub fn peek_next(&self) -> Option<(&K, &mut V)> {
        let (k, v) = self.0.peek_next()?;
        Some((&*k, v))
    }

    /// Returns references to the previous key/value pair.
    pub fn peek_prev(&self) -> Option<(&K, &mut V)> {
        let (k, v) = self.0.peek_prev()?;
        Some((&*k, v))
    }

    /// Converts the cursor into a CursorMutKey, which allows mutating the key of elements in the tree.
    /// # Safety
    ///
    /// Keys must be unique and in sorted order.
    pub unsafe fn with_mutable_key(self) -> CursorMutKey<'a, K, V, B> {
        self.0
    }

    /// Returns a read-only cursor pointing to the same location as the CursorMut.
    pub fn as_cursor(&self) -> Cursor<'_, K, V, B> {
        self.0.as_cursor()
    }
}

/// Cursor that allows mutation of map keys, returned by [CursorMut::with_mutable_key].
pub struct CursorMutKey<'a, K, V, const B: usize> {
    map: *mut BTreeMap<K, V, B>,
    leaf: Option<*mut Leaf<K, V, B>>,
    index: usize,
    stack: ArrayVec<(*mut NonLeaf<K, V, B>, usize), 10>,
    _pd: PhantomData<&'a mut BTreeMap<K, V, B>>,
}

unsafe impl<'a, K, V, const B: usize> Send for CursorMutKey<'a, K, V, B> {}
unsafe impl<'a, K, V, const B: usize> Sync for CursorMutKey<'a, K, V, B> {}

impl<'a, K, V, const B: usize> CursorMutKey<'a, K, V, B> {
    fn make(map: *mut BTreeMap<K, V, B>) -> Self {
        Self {
            map,
            leaf: None,
            index: 0,
            stack: ArrayVec::new(),
            _pd: PhantomData,
        }
    }

    fn push_lower<Q>(&mut self, tree: &mut Tree<K, V, B>, bound: Bound<&Q>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match tree {
            Tree::L(leaf) => {
                self.index = leaf.get_lower(bound);
                self.leaf = Some(leaf);
            }
            Tree::NL(x) => {
                let ix = x.get_lower(bound);
                self.stack.push((x, ix));
                self.push_lower(x.c.ixm(ix), bound);
            }
        }
    }

    fn push_upper<Q>(&mut self, tree: &mut Tree<K, V, B>, bound: Bound<&Q>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match tree {
            Tree::L(leaf) => {
                self.index = leaf.get_upper(bound);
                self.leaf = Some(leaf);
            }
            Tree::NL(x) => {
                let ix = x.get_upper(bound);
                self.stack.push((x, ix));
                self.push_upper(x.c.ixm(ix), bound);
            }
        }
    }

    fn push(&mut self, tree: &mut Tree<K, V, B>) {
        match tree {
            Tree::L(leaf) => {
                self.index = 0;
                self.leaf = Some(leaf);
            }
            Tree::NL(x) => {
                self.stack.push((x, 0));
                self.push(x.c.ixm(0));
            }
        }
    }

    fn push_back(&mut self, tree: &mut Tree<K, V, B>) {
        match tree {
            Tree::L(leaf) => {
                self.index = leaf.0.len();
                self.leaf = Some(leaf);
            }
            Tree::NL(x) => {
                let ix = x.v.len();
                self.stack.push((x, ix));
                self.push_back(x.c.ixm(ix));
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
                let (med, right) = (*leaf).split();
                let right = Tree::L(Leaf(right));
                let r = if self.index > B / 2 { 1 } else { 0 };
                self.index -= r * (B / 2 + 1);
                let t = self.split(med, right, r);
                leaf = (*t).leaf();
                self.leaf = Some(leaf);
            }
            (*leaf).0.insert(self.index, (key, value));
        }
    }

    fn split(&mut self, med: (K, V), tree: Tree<K, V, B>, r: usize) -> *mut Tree<K, V, B> {
        unsafe {
            if let Some((mut nl, mut ix)) = self.stack.pop() {
                if (*nl).full() {
                    let (med, tree) = (*nl).split();
                    let r = if ix > B / 2 { 1 } else { 0 };
                    ix -= r * (B / 2 + 1);
                    let t = self.split(med, tree, r);
                    nl = (*t).nonleaf();
                }
                (*nl).v.insert(ix, med);
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
                while let Some((nl, mut ix)) = self.stack.pop() {
                    if ix < (*nl).v.len() {
                        let kv;
                        if let Some(rep) = (*nl).c.ixm(ix).pop_last() {
                            kv = std::mem::replace((*nl).v.ixm(ix), rep);
                            ix += 1;
                        } else {
                            kv = (*nl).v.remove(ix);
                            (*nl).c.remove(ix);
                        }
                        self.stack.push((nl, ix));
                        self.push((*nl).c.ixm(ix));
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
                while let Some((nl, mut ix)) = self.stack.pop() {
                    if ix < (*nl).v.len() {
                        let kv = (*nl).v.ixm(ix);
                        ix += 1;
                        self.stack.push((nl, ix));
                        self.push((*nl).c.ixm(ix));
                        return Some((&mut kv.0, &mut kv.1));
                    }
                }
                None
            } else {
                let kv = (*leaf).0.ixm(self.index);
                self.index += 1;
                Some((&mut kv.0, &mut kv.1))
            }
        }
    }

    /// Move the cursor back, returns references to the key and value of the element that it moved over.
    pub fn prev(&mut self) -> Option<(&mut K, &mut V)> {
        unsafe {
            if self.index == 0 {
                while let Some((nl, mut ix)) = self.stack.pop() {
                    if ix > 0 {
                        ix -= 1;
                        let kv = (*nl).v.ixm(ix);
                        self.stack.push((nl, ix));
                        self.push_back((*nl).c.ixm(ix));
                        return Some((&mut kv.0, &mut kv.1));
                    }
                }
                None
            } else {
                let leaf = self.leaf.unwrap_unchecked();
                self.index -= 1;
                let kv = (*leaf).0.ixm(self.index);
                Some((&mut kv.0, &mut kv.1))
            }
        }
    }

    /// Returns references to the next key/value pair.
    pub fn peek_next(&self) -> Option<(&mut K, &mut V)> {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            if self.index == (*leaf).0.len() {
                for (nl, ix) in self.stack.iter().rev() {
                    if *ix < (**nl).v.len() {
                        let kv = (**nl).v.ixm(*ix);
                        return Some((&mut kv.0, &mut kv.1));
                    }
                }
                None
            } else {
                let kv = (*leaf).0.ixm(self.index);
                Some((&mut kv.0, &mut kv.1))
            }
        }
    }
    /// Returns references to the previous key/value pair.
    pub fn peek_prev(&self) -> Option<(&mut K, &mut V)> {
        unsafe {
            if self.index == 0 {
                for (nl, ix) in self.stack.iter().rev() {
                    if *ix > 0 {
                        let kv = (**nl).v.ixm(*ix - 1);
                        return Some((&mut kv.0, &mut kv.1));
                    }
                }
                None
            } else {
                let leaf = self.leaf.unwrap_unchecked();
                let kv = (*leaf).0.ixm(self.index - 1);
                Some((&mut kv.0, &mut kv.1))
            }
        }
    }

    /// Returns a read-only cursor pointing to the same location as the CursorMutKey.
    pub fn as_cursor(&self) -> Cursor<'_, K, V, B> {
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
}

/// Cursor returned by [BTreeMap::lower_bound], [BTreeMap::upper_bound].
#[derive(Debug, Clone)]
pub struct Cursor<'a, K, V, const B: usize> {
    leaf: Option<*const Leaf<K, V, B>>, // Maybe can use orderinary reference rather than pointer.
    index: usize,
    stack: ArrayVec<(*const NonLeaf<K, V, B>, usize), 10>,
    _pd: PhantomData<&'a BTreeMap<K, V, B>>,
}

unsafe impl<'a, K, V, const B: usize> Send for Cursor<'a, K, V, B> {}
unsafe impl<'a, K, V, const B: usize> Sync for Cursor<'a, K, V, B> {}

impl<'a, K, V, const B: usize> Cursor<'a, K, V, B> {
    fn make() -> Self {
        Self {
            leaf: None,
            index: 0,
            stack: ArrayVec::new(),
            _pd: PhantomData,
        }
    }

    fn lower_bound<Q>(bt: &BTreeMap<K, V, B>, bound: Bound<&Q>) -> Self
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let mut s = Self::make();
        s.push_lower(&bt.tree, bound);
        s
    }

    fn upper_bound<Q>(bt: &BTreeMap<K, V, B>, bound: Bound<&Q>) -> Self
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let mut s = Self::make();
        s.push_upper(&bt.tree, bound);
        s
    }

    fn push_lower<Q>(&mut self, tree: &Tree<K, V, B>, bound: Bound<&Q>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match tree {
            Tree::L(leaf) => {
                self.leaf = Some(leaf);
                self.index = leaf.get_lower(bound);
            }
            Tree::NL(nl) => {
                let ix = nl.get_lower(bound);
                self.stack.push((nl, ix));
                let c = &nl.c[ix];
                self.push_lower(c, bound);
            }
        }
    }

    fn push_upper<Q>(&mut self, tree: &Tree<K, V, B>, bound: Bound<&Q>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match tree {
            Tree::L(leaf) => {
                self.leaf = Some(leaf);
                self.index = leaf.get_upper(bound);
            }
            Tree::NL(nl) => {
                let ix = nl.get_upper(bound);
                self.stack.push((nl, ix));
                let c = &nl.c[ix];
                self.push_upper(c, bound);
            }
        }
    }

    fn push(&mut self, tree: &Tree<K, V, B>) {
        match tree {
            Tree::L(leaf) => {
                self.leaf = Some(leaf);
                self.index = 0;
            }
            Tree::NL(x) => {
                self.stack.push((x, 0));
                let c = &x.c[0];
                self.push(c);
            }
        }
    }

    fn push_back(&mut self, tree: &Tree<K, V, B>) {
        match tree {
            Tree::L(leaf) => {
                self.leaf = Some(leaf);
                self.index = leaf.0.len();
            }
            Tree::NL(x) => {
                let ix = x.v.len();
                self.stack.push((x, ix));
                let c = &x.c[ix];
                self.push_back(c);
            }
        }
    }

    /// Advance the cursor, returns references to the key and value of the element that it moved over.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<(&K, &V)> {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            if self.index == (*leaf).0.len() {
                loop {
                    if let Some((nl, mut ix)) = self.stack.pop() {
                        if ix < (*nl).v.len() {
                            let kv: *const (K, V) = (*nl).v.ix(ix);
                            ix += 1;
                            let ct = (*nl).c.ix(ix);
                            self.stack.push((nl, ix));
                            self.push(ct);
                            return Some((&(*kv).0, &(*kv).1));
                        }
                    } else {
                        return None;
                    }
                }
            } else {
                let kv: *const (K, V) = (*leaf).0.ix(self.index);
                self.index += 1;
                Some((&(*kv).0, &(*kv).1))
            }
        }
    }

    /// Move the cursor back, returns references to the key and value of the element that it moved over.
    #[allow(clippy::should_implement_trait)]
    pub fn prev(&mut self) -> Option<(&K, &V)> {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            if self.index == 0 {
                loop {
                    if let Some((nl, mut ix)) = self.stack.pop() {
                        if ix > 0 {
                            ix -= 1;
                            let kv: *const (K, V) = (*nl).v.ix(ix);
                            let ct = (*nl).c.ix(ix);
                            self.stack.push((nl, ix));
                            self.push_back(ct);
                            return Some((&(*kv).0, &(*kv).1));
                        }
                    } else {
                        return None;
                    }
                }
            } else {
                self.index -= 1;
                let kv: *const (K, V) = (*leaf).0.ix(self.index);
                Some((&(*kv).0, &(*kv).1))
            }
        }
    }

    /// Returns references to the next key/value pair.
    pub fn peek_next(&self) -> Option<(&K, &V)> {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            if self.index == (*leaf).0.len() {
                for (nl, ix) in self.stack.iter().rev() {
                    if *ix < (**nl).v.len() {
                        let kv: *const (K, V) = (**nl).v.ix(*ix);
                        return Some((&(*kv).0, &(*kv).1));
                    }
                }
                None
            } else {
                let kv: *const (K, V) = (*leaf).0.ix(self.index);
                Some((&(*kv).0, &(*kv).1))
            }
        }
    }
    /// Returns references to the previous key/value pair.
    pub fn peek_prev(&self) -> Option<(&K, &V)> {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            if self.index == 0 {
                for (nl, ix) in self.stack.iter().rev() {
                    if *ix > 0 {
                        let kv: *const (K, V) = (**nl).v.ix(*ix - 1);
                        return Some((&(*kv).0, &(*kv).1));
                    }
                }
                None
            } else {
                let kv: *const (K, V) = (*leaf).0.ix(self.index - 1);
                Some((&(*kv).0, &(*kv).1))
            }
        }
    }
}