use std::{
    borrow::Borrow,
    cmp::Ordering,
    // fmt,
    // fmt::Debug,
    iter::FusedIterator,
    marker::PhantomData,
    mem,
    mem::MaybeUninit,
    ops::{Bound /*, RangeBounds*/},
    ptr,
};

/// `BTreeMap` similar to [`std::collections::BTreeMap`] where the node capacity (N) can be specified.
///
/// N must be an odd number, at least 11, less than 255, good values to try might be 15, 23, 31, 39, 47.
///
/// M must be one more than N ( once const generics are fully implemented it can go away entirely ).
///
/// General guide to implementation:
///
/// To minimise storage use, child lengths are stored as a byte in the parent.
///
/// The [`Entry`] API is implemented using [`CursorMut`].
///
/// [`CursorMut`] is implemented using [`CursorMutKey`] which has a stack of raw pointer/index pairs
/// to keep track of non-leaf positions.

pub struct BTreeMap<K, V, const N: usize, const M: usize> {
    len: usize,
    clen: u8,
    tree: Tree<K, V, N, M>,
}

impl<K, V, const N: usize, const M: usize> Drop for BTreeMap<K, V, N, M> {
    fn drop(&mut self) {
        match &mut self.tree {
            Tree::L(leaf) => leaf.free(&mut self.clen),
            Tree::NL(nl) => nl.free(&mut self.clen),
        }
    }
}

impl<K, V, const N: usize, const M: usize> BTreeMap<K, V, N, M> {
    /// Returns a new, empty map.
    pub fn new() -> Self {
        Self {
            len: 0,
            clen: 0,
            tree: Tree::default(),
        }
    }

    fn new_root(&mut self, split: Split<K, V, N, M>) {
        self.tree.new_root(self.clen, split);
        self.clen = 1;
    }

    /// Clear the map.
    pub fn clear(&mut self) {
        *self = Self::new();
    }

    /// Print node lengths ( only goes down 1 level ).
    pub fn print_clen(&self) {
        match &self.tree {
            Tree::L(_) => println!("clen={}", self.clen),
            Tree::NL(nl) => nl.print_clen(self.clen as usize),
        };
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
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, N, M>
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
    pub fn first_entry(&mut self) -> Option<OccupiedEntry<'_, K, V, N, M>>
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
    pub fn last_entry(&mut self) -> Option<OccupiedEntry<'_, K, V, N, M>>
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
        };
        match &mut self.tree {
            Tree::L(leaf) => leaf.insert_kv(&mut self.clen, key, &mut x),
            Tree::NL(nl) => nl.insert_kv(&mut self.clen, key, &mut x),
        }
        if let Some(split) = x.split {
            self.new_root(split);
        }
        self.len += usize::from(x.value.is_none());
        x.value
    }

    /// Tries to insert a key-value pair into the map, and returns
    /// a mutable reference to the value in the entry.
    ///
    /// If the map already had this key present, nothing is updated, and
    /// an error containing the occupied entry and the value is returned.
    pub fn try_insert(&mut self, key: K, value: V) -> Result<&mut V, OccupiedError<'_, K, V, N, M>>
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

    /// Remove first key-value pair from map.
    pub fn pop_first(&mut self) -> Option<(K, V)> {
        let n = &mut self.clen;
        let result = match &mut self.tree {
            Tree::L(leaf) => leaf.pop_first(n),
            Tree::NL(nl) => nl.pop_first(n),
        };
        self.len -= usize::from(result.is_some());
        result
    }

    /// Remove last key-value pair from map.
    pub fn pop_last(&mut self) -> Option<(K, V)> {
        let n = &mut self.clen;
        let result = match &mut self.tree {
            Tree::L(leaf) => leaf.pop_last(n),
            Tree::NL(nl) => nl.pop_last(n),
        };
        self.len -= usize::from(result.is_some());
        result
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
        let n = &mut self.clen;
        let result = match &mut self.tree {
            Tree::L(leaf) => leaf.remove_entry(n, key),
            Tree::NL(nl) => nl.remove_entry(n, key),
        };
        self.len -= usize::from(result.is_some());
        result
    }

    /// Get reference to the value corresponding to the key.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let n = self.clen as usize;
        match &self.tree {
            Tree::L(leaf) => leaf.get(n, key),
            Tree::NL(nl) => nl.get(n, key),
        }
    }

    /// Get references to the corresponding key and value.
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let n = self.clen as usize;
        match &self.tree {
            Tree::L(leaf) => leaf.get_key_value(n, key),
            Tree::NL(nl) => nl.get_key_value(n, key),
        }
    }

    /// Get mutable reference to the value corresponding to the key.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let n = self.clen as usize;
        match &mut self.tree {
            Tree::L(leaf) => leaf.get_mut(n, key),
            Tree::NL(nl) => nl.get_mut(n, key),
        }
    }

    /// Get a cursor positioned just after bound that permits map mutation.
    pub fn lower_bound_mut<Q>(&mut self, bound: Bound<&Q>) -> CursorMut<'_, K, V, N, M>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        CursorMut::lower_bound(self, bound)
    }

    /// Get a cursor positioned just before bound that permits map mutation.
    pub fn upper_bound_mut<Q>(&mut self, bound: Bound<&Q>) -> CursorMut<'_, K, V, N, M>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        CursorMut::upper_bound(self, bound)
    }

    /// Get cursor positioned just after bound.
    #[must_use]
    pub fn lower_bound<Q>(&self, bound: Bound<&Q>) -> Cursor<'_, K, V, N, M>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        Cursor::lower_bound(self, bound)
    }

    /// Get cursor positioned just before bound.
    #[must_use]
    pub fn upper_bound<Q>(&self, bound: Bound<&Q>) -> Cursor<'_, K, V, N, M>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        Cursor::upper_bound(self, bound)
    }

    /// Get iterator of references to key-value pairs.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, K, V, N, M> {
        Iter {
            len: self.len,
            inner: self.tree.iter(&self.clen),
        }
    }

    /// Get iterator of mutable references to key-value pairs.
    #[must_use]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V, N, M> {
        IterMut {
            len: self.len,
            inner: self.tree.iter_mut(&self.clen),
        }
    }
} // end impl BTreeMap

impl<K, V, const N: usize, const M: usize> Default for BTreeMap<K, V, N, M> {
    fn default() -> Self {
        Self::new()
    }
}

type Split<K, V, const N: usize, const M: usize> = ((K, V), Tree<K, V, N, M>, usize);

enum Tree<K, V, const N: usize, const M: usize> {
    L(Box<Leaf<K, V, N>>),
    NL(Box<NonLeaf<K, V, N, M>>),
}

impl<K, V, const N: usize, const M: usize> Default for Tree<K, V, N, M> {
    fn default() -> Self {
        Self::L(Box::new(Leaf::new()))
    }
}

impl<K, V, const N: usize, const M: usize> Tree<K, V, N, M> {
    fn new_root(&mut self, left_len: u8, ((k, v), right, rlen): Split<K, V, N, M>) {
        let child_is_leaf = match right {
            Tree::L(_) => true,
            Tree::NL(_) => false,
        };
        let mut nl = Box::new(NonLeaf::new(child_is_leaf));
        nl.clen[0] = left_len;
        nl.clen[1] = rlen as u8;
        nl.push_child(0, std::mem::take(self));
        nl.leaf.push(0, k, v);
        nl.push_child(1, right);
        *self = Tree::NL(nl);
    }
    fn leaf(self) -> Box<Leaf<K, V, N>> {
        match self {
            Tree::L(x) => x,
            _ => panic!(),
        }
    }
    fn non_leaf(self) -> Box<NonLeaf<K, V, N, M>> {
        match self {
            Tree::NL(x) => x,
            _ => panic!(),
        }
    }
    fn non_leaf_ptr(&mut self) -> &mut NonLeaf<K, V, N, M> {
        match self {
            Tree::NL(x) => x,
            _ => panic!(),
        }
    }
    fn iter<'a>(&'a self, clen: &'a u8) -> Range<'a, K, V, N, M> {
        let mut x = Range::new();
        let tr = match self {
            Tree::L(x) => TreeRef::L(x, clen),
            Tree::NL(x) => TreeRef::NL(x, clen),
        };
        x.push_tree(tr, true);
        x
    }
    fn iter_mut<'a>(&'a mut self, clen: &'a u8) -> RangeMut<'a, K, V, N, M> {
        let mut x = RangeMut::new();
        let tr = match self {
            Tree::L(x) => TreeRefMut::L(x, clen),
            Tree::NL(x) => TreeRefMut::NL(x, clen),
        };
        x.push_tree(tr, true);
        x
    }
} // end impl Tree

unsafe fn ix<T>(p: *const [T], ix: usize) -> *const T {
    p.cast::<T>().add(ix)
}

unsafe fn ixm<T>(p: *mut [T], ix: usize) -> *mut T {
    p.cast::<T>().add(ix)
}

struct InsertCtx<K, V, const N: usize, const M: usize> {
    value: Option<V>,
    split: Option<Split<K, V, N, M>>,
}

struct Leaf<K, V, const N: usize> {
    keys: MaybeUninit<[K; N]>,
    vals: MaybeUninit<[V; N]>,
}

impl<K, V, const N: usize> Leaf<K, V, N> {
    fn new() -> Self {
        Self {
            keys: MaybeUninit::uninit(),
            vals: MaybeUninit::uninit(),
        }
    }

    /// Cannot implement Drop as len needs to be a parameter.
    /// Instead this function is called to drop any stored key-value pairs.
    fn free(&mut self, len: &mut u8) {
        while *len > 0 {
            self.pop_last(len);
        }
    }

    pub fn get<Q>(&self, len: usize, key: &Q) -> Option<&V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self.search(len, |x| x.borrow().cmp(key)) {
            Ok(i) => Some(self.ixv(i)),
            Err(_) => None,
        }
    }

    pub fn get_key_value<Q>(&self, len: usize, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self.search(len, |x| x.borrow().cmp(key)) {
            Ok(i) => Some(self.kv(i)),
            Err(_) => None,
        }
    }

    pub fn get_mut<Q>(&mut self, len: usize, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self.search(len, |x| x.borrow().cmp(key)) {
            Ok(i) => Some(self.ixvm(i)),
            Err(_) => None,
        }
    }

    fn insert_kv<const M: usize>(&mut self, len: &mut u8, key: K, x: &mut InsertCtx<K, V, N, M>)
    where
        K: Ord,
    {
        let mut i = match self.search(*len as usize, |x| x.borrow().cmp(&key)) {
            Ok(i) => {
                let value = x.value.take().unwrap();
                x.value = Some(std::mem::replace(self.ixvm(i), value));
                *self.ixkm(i) = key;
                return;
            }
            Err(i) => i,
        };
        let value = x.value.take().unwrap();
        if *len as usize == N {
            // Leaf is full.
            let (med, mut right) = self.split(len);
            let mut rlen = N / 2;
            if i > N / 2 {
                i -= N / 2 + 1;
                right.insert(N / 2, i, key, value);
                rlen += 1;
            } else {
                self.insert(*len as usize, i, key, value);
                *len += 1;
            }
            let right = Tree::L(right);
            x.split = Some((med, right, rlen));
        } else {
            self.insert(*len as usize, i, key, value);
            *len += 1;
        }
    }

    pub fn remove_entry<Q>(&mut self, len: &mut u8, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self.search(*len as usize, |x| x.borrow().cmp(key)) {
            Ok(i) => Some(self.remove(len, i)),
            Err(_i) => None,
        }
    }

    /// Move keys and values ( used when splitting node ).
    fn mov(&mut self, at: usize, len: usize, to: &mut Self) {
        unsafe {
            let kp = ix(self.keys.as_ptr(), at);
            let rkp = ixm(to.keys.as_mut_ptr(), 0);
            ptr::copy_nonoverlapping(kp, rkp, len);
            let vp = ix(self.vals.as_ptr(), at);
            let rvp = ixm(to.vals.as_mut_ptr(), 0);
            ptr::copy_nonoverlapping(vp, rvp, len);
        }
    }

    /// Split leaf. Each half will have length N/2
    fn split(&mut self, len: &mut u8) -> ((K, V), Box<Self>) {
        assert!(*len as usize == N);
        let mut right = Box::new(Self::new());
        self.mov(N / 2 + 1, N / 2, &mut right);
        *len = (N / 2 + 1) as u8;
        let med = self.pop_last(len).unwrap();
        (med, right)
    }

    fn insert(&mut self, len: usize, at: usize, key: K, val: V) {
        assert!(at <= len && len < N);
        let n = len - at;
        unsafe {
            let kp = ixm(self.keys.as_mut_ptr(), at);
            let vp = ixm(self.vals.as_mut_ptr(), at);
            if n > 0 {
                ptr::copy(kp, kp.add(1), n);
                ptr::copy(vp, vp.add(1), n);
            }
            kp.write(key);
            vp.write(val);
        }
    }

    fn remove(&mut self, len: &mut u8, at: usize) -> (K, V) {
        assert!(at < *len as usize);
        *len -= 1;
        unsafe {
            let kp = ixm(self.keys.as_mut_ptr(), at);
            let vp = ixm(self.vals.as_mut_ptr(), at);
            let result = (kp.read(), vp.read());
            let n = *len as usize - at;
            if n > 0 {
                ptr::copy(kp.add(1), kp, n);
                ptr::copy(vp.add(1), vp, n);
            }
            result
        }
    }

    fn replace(&mut self, at: usize, kv: (K, V)) -> (K, V) {
        unsafe {
            let kp = ixm(self.keys.as_mut_ptr(), at);
            let vp = ixm(self.vals.as_mut_ptr(), at);
            let k = mem::replace(&mut *kp, kv.0);
            let v = mem::replace(&mut *vp, kv.1);
            (k, v)
        }
    }

    fn push(&mut self, len: usize, key: K, val: V) {
        assert!(len < N);
        unsafe {
            let kp = ixm(self.keys.as_mut_ptr(), len);
            let vp = ixm(self.vals.as_mut_ptr(), len);
            kp.write(key);
            vp.write(val);
        }
    }

    fn pop_first(&mut self, len: &mut u8) -> Option<(K, V)> {
        if *len == 0 {
            None
        } else {
            Some(self.remove(len, 0))
        }
    }

    fn pop_last(&mut self, len: &mut u8) -> Option<(K, V)> {
        if *len == 0 {
            None
        } else {
            *len -= 1;
            let len = *len as usize;
            unsafe {
                let kp = ix(self.keys.as_ptr(), len);
                let vp = ix(self.vals.as_ptr(), len);
                Some((kp.read(), vp.read()))
            }
        }
    }

    fn iter(&self, len: &u8) -> LeafIter<K, V, N> {
        LeafIter {
            leaf: self,
            fwd: 0,
            bck: *len as usize,
        }
    }

    fn iter_mut(&mut self, len: &u8) -> LeafIterMut<K, V, N> {
        LeafIterMut {
            leaf: self,
            fwd: 0,
            bck: *len as usize,
        }
    }

    /// Get reference to ith key.
    #[inline]
    fn ixk(&self, i: usize) -> &K {
        unsafe { &*ix(self.keys.as_ptr(), i) }
    }

    /// Get mutable reference to ith key.
    #[inline]
    fn ixkm(&mut self, i: usize) -> &mut K {
        unsafe { &mut *ixm(self.keys.as_mut_ptr(), i) }
    }

    /// Get reference to ith value.
    #[inline]
    fn ixv(&self, i: usize) -> &V {
        unsafe { &*ix(self.vals.as_ptr(), i) }
    }

    /// Get mutable reference to ith value.
    #[inline]
    fn ixvm(&mut self, i: usize) -> &mut V {
        unsafe { &mut *ixm(self.vals.as_mut_ptr(), i) }
    }

    /// Get references to ith key and ith value.
    #[inline]
    fn kv(&self, i: usize) -> (&K, &V) {
        (self.ixk(i), self.ixv(i))
    }

    /// Get mutable references to ith key and ith value.
    #[inline]
    fn kvm(&mut self, i: usize) -> (&mut K, &mut V) {
        unsafe {
            (
                &mut *ixm(self.keys.as_mut_ptr(), i),
                &mut *ixm(self.vals.as_mut_ptr(), i),
            )
        }
    }

    fn search<F>(&self, len: usize, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&K) -> Ordering,
    {
        let (mut i, mut j) = (0, len);
        while i < j {
            let m = (i + j) / 2;
            match f(self.ixk(m)) {
                Ordering::Equal => {
                    return Ok(m);
                }
                Ordering::Less => i = m + 1,
                Ordering::Greater => j = m,
            }
        }
        Err(i)
    }

    fn get_lower<Q>(&self, len: usize, bound: Bound<&Q>) -> usize
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match bound {
            Bound::Unbounded => 0,
            Bound::Included(k) => match self.search(len, |a| a.borrow().cmp(k)) {
                Ok(x) | Err(x) => x,
            },
            Bound::Excluded(k) => match self.search(len, |a| a.borrow().cmp(k)) {
                Ok(x) => x + 1,
                Err(x) => x,
            },
        }
    }

    fn get_upper<Q>(&self, len: usize, bound: Bound<&Q>) -> usize
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match bound {
            Bound::Unbounded => len,
            Bound::Included(k) => match self.search(len, |a| a.borrow().cmp(k)) {
                Ok(x) => x + 1,
                Err(x) => x,
            },
            Bound::Excluded(k) => match self.search(len, |a| a.borrow().cmp(k)) {
                Ok(x) | Err(x) => x,
            },
        }
    }
} // end impl Leaf

struct LeafIter<'a, K, V, const N: usize> {
    leaf: &'a Leaf<K, V, N>,
    fwd: usize,
    bck: usize,
}

impl<'a, K, V, const N: usize> LeafIter<'a, K, V, N> {
    fn len(&self) -> usize {
        self.bck - self.fwd
    }
}

impl<'a, K, V, const N: usize> Iterator for LeafIter<'a, K, V, N> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        if self.fwd == self.bck {
            None
        } else {
            unsafe {
                let kp = ix(self.leaf.keys.as_ptr(), self.fwd);
                let vp = ix(self.leaf.vals.as_ptr(), self.fwd);
                self.fwd += 1;
                Some((&*kp, &*vp))
            }
        }
    }
}

impl<'a, K, V, const N: usize> DoubleEndedIterator for LeafIter<'a, K, V, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.fwd == self.bck {
            None
        } else {
            unsafe {
                self.bck -= 1;
                let kp = ix(self.leaf.keys.as_ptr(), self.bck);
                let vp = ix(self.leaf.vals.as_ptr(), self.bck);
                Some((&*kp, &*vp))
            }
        }
    }
}

/// ...
pub struct LeafIterMut<'a, K, V, const N: usize> {
    leaf: &'a mut Leaf<K, V, N>,
    fwd: usize,
    bck: usize,
}

impl<'a, K, V, const N: usize> LeafIterMut<'a, K, V, N> {
    fn len(&self) -> usize {
        self.bck - self.fwd
    }
}

impl<'a, K, V, const N: usize> Iterator for LeafIterMut<'a, K, V, N> {
    type Item = (&'a K, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        if self.fwd == self.bck {
            None
        } else {
            unsafe {
                let kp = ix(self.leaf.keys.as_ptr(), self.fwd);
                let vp = ixm(self.leaf.vals.as_mut_ptr(), self.fwd);
                self.fwd += 1;
                Some((&*kp, &mut *vp))
            }
        }
    }
}

impl<'a, K, V, const N: usize> DoubleEndedIterator for LeafIterMut<'a, K, V, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.fwd == self.bck {
            None
        } else {
            unsafe {
                self.bck -= 1;
                let kp = ix(self.leaf.keys.as_ptr(), self.bck);
                let vp = ixm(self.leaf.vals.as_mut_ptr(), self.bck);
                Some((&*kp, &mut *vp))
            }
        }
    }
}

/// ...
pub struct LeafIntoIter<K, V, const N: usize> {
    leaf: Leaf<K, V, N>,
    fwd: usize,
    bck: usize,
}

impl<K, V, const N: usize> Drop for LeafIntoIter<K, V, N> {
    fn drop(&mut self) {
        while self.fwd != self.bck {
            self.next();
        }
    }
}

impl<K, V, const N: usize> Iterator for LeafIntoIter<K, V, N> {
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        if self.fwd == self.bck {
            None
        } else {
            unsafe {
                let kp = ix(self.leaf.keys.as_ptr(), self.fwd);
                let vp = ix(self.leaf.vals.as_ptr(), self.fwd);
                self.fwd += 1;
                Some((kp.read(), vp.read()))
            }
        }
    }
}
impl<K, V, const N: usize> DoubleEndedIterator for LeafIntoIter<K, V, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.fwd == self.bck {
            None
        } else {
            unsafe {
                self.bck -= 1;
                let kp = ix(self.leaf.keys.as_ptr(), self.bck);
                let vp = ix(self.leaf.vals.as_ptr(), self.bck);
                Some((kp.read(), vp.read()))
            }
        }
    }
}

enum ChildIter<'a, K, V, const N: usize, const M: usize> {
    L(
        std::slice::Iter<'a, Box<Leaf<K, V, N>>>,
        std::slice::Iter<'a, u8>,
    ),
    NL(
        std::slice::Iter<'a, Box<NonLeaf<K, V, N, M>>>,
        std::slice::Iter<'a, u8>,
    ),
}

enum TreeRef<'a, K, V, const N: usize, const M: usize> {
    L(&'a Leaf<K, V, N>, &'a u8),
    NL(&'a NonLeaf<K, V, N, M>, &'a u8),
}

enum TreeRefMut<'a, K, V, const N: usize, const M: usize> {
    L(&'a mut Leaf<K, V, N>, &'a u8),
    NL(&'a mut NonLeaf<K, V, N, M>, &'a u8),
}

impl<'a, K, V, const N: usize, const M: usize> ChildIter<'a, K, V, N, M> {
    fn len(&self) -> usize {
        match self {
            ChildIter::L(_, n) => n.len(),
            ChildIter::NL(_, n) => n.len(),
        }
    }
}
impl<'a, K, V, const N: usize, const M: usize> Iterator for ChildIter<'a, K, V, N, M> {
    type Item = TreeRef<'a, K, V, N, M>;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ChildIter::L(x, n) => Some(TreeRef::L(x.next()?, n.next()?)),
            ChildIter::NL(x, n) => Some(TreeRef::NL(x.next()?, n.next()?)),
        }
    }
}
impl<'a, K, V, const N: usize, const M: usize> DoubleEndedIterator for ChildIter<'a, K, V, N, M> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            ChildIter::L(x, n) => Some(TreeRef::L(x.next_back()?, n.next_back()?)),
            ChildIter::NL(x, n) => Some(TreeRef::NL(x.next_back()?, n.next_back()?)),
        }
    }
}

enum ChildIterMut<'a, K, V, const N: usize, const M: usize> {
    L(
        std::slice::IterMut<'a, Box<Leaf<K, V, N>>>,
        std::slice::Iter<'a, u8>,
    ),
    NL(
        std::slice::IterMut<'a, Box<NonLeaf<K, V, N, M>>>,
        std::slice::Iter<'a, u8>,
    ),
}

impl<'a, K, V, const N: usize, const M: usize> ChildIterMut<'a, K, V, N, M> {
    fn len(&self) -> usize {
        match self {
            Self::L(_, n) => n.len(),
            Self::NL(_, n) => n.len(),
        }
    }
}
impl<'a, K, V, const N: usize, const M: usize> Iterator for ChildIterMut<'a, K, V, N, M> {
    type Item = TreeRefMut<'a, K, V, N, M>;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::L(x, n) => Some(TreeRefMut::L(x.next()?, n.next()?)),
            Self::NL(x, n) => Some(TreeRefMut::NL(x.next()?, n.next()?)),
        }
    }
}
impl<'a, K, V, const N: usize, const M: usize> DoubleEndedIterator
    for ChildIterMut<'a, K, V, N, M>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            Self::L(x, n) => Some(TreeRefMut::L(x.next_back()?, n.next_back()?)),
            Self::NL(x, n) => Some(TreeRefMut::NL(x.next_back()?, n.next_back()?)),
        }
    }
}

//#[derive(Clone, Debug)]
struct Stk<'a, K, V, const N: usize, const M: usize> {
    v: LeafIter<'a, K, V, N>,
    c: ChildIter<'a, K, V, N, M>,
}

/// Iterator returned by [`BTreeMap::range`].
//#[derive(Clone, Debug, Default)]
pub struct Range<'a, K, V, const N: usize, const M: usize> {
    fwd_leaf: Option<LeafIter<'a, K, V, N>>,
    bck_leaf: Option<LeafIter<'a, K, V, N>>,
    fwd_stk: StkVec<Stk<'a, K, V, N, M>>,
    bck_stk: StkVec<Stk<'a, K, V, N, M>>,
}
impl<'a, K, V, const N: usize, const M: usize> Range<'a, K, V, N, M> {
    fn new() -> Self {
        Self {
            fwd_leaf: None,
            bck_leaf: None,
            fwd_stk: StkVec::new(),
            bck_stk: StkVec::new(),
        }
    }

    fn push_tree(&mut self, tree: TreeRef<'a, K, V, N, M>, both: bool) {
        match tree {
            TreeRef::L(leaf, len) => {
                self.fwd_leaf = Some(leaf.iter(len));
            }
            TreeRef::NL(nl, len) => {
                let (v, mut c) = nl.iter(len);
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

    fn push_tree_back(&mut self, tree: TreeRef<'a, K, V, N, M>) {
        match tree {
            TreeRef::L(leaf, len) => {
                self.bck_leaf = Some(leaf.iter(len));
            }
            TreeRef::NL(nl, len) => {
                let (v, mut c) = nl.iter(len);
                let ct_back = c.next_back();
                self.bck_stk.push(Stk { v, c });
                if let Some(ct_back) = ct_back {
                    self.push_tree_back(ct_back);
                }
            }
        }
    }
    fn steal_bck(&mut self) -> StealResult<'a, K, V, N, M> {
        for s in &mut self.bck_stk {
            if s.v.len() > s.c.len() {
                return StealResult::KV(s.v.next().unwrap());
            } else if let Some(ct) = s.c.next() {
                return StealResult::CT(ct);
            }
        }
        StealResult::Nothing
    }
    fn steal_fwd(&mut self) -> StealResult<'a, K, V, N, M> {
        for s in &mut self.fwd_stk {
            if s.v.len() > s.c.len() {
                return StealResult::KV(s.v.next_back().unwrap());
            } else if let Some(ct) = s.c.next_back() {
                return StealResult::CT(ct);
            }
        }
        StealResult::Nothing
    }
}
impl<'a, K, V, const N: usize, const M: usize> Iterator for Range<'a, K, V, N, M> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(f) = &mut self.fwd_leaf {
                if let Some(x) = f.next() {
                    return Some(x);
                }
                self.fwd_leaf = None;
            } else if let Some(s) = self.fwd_stk.last_mut() {
                if let Some(kv) = s.v.next() {
                    if let Some(ct) = s.c.next() {
                        self.push_tree(ct, false);
                    }
                    return Some(kv);
                }
                self.fwd_stk.pop();
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
                            }
                            self.bck_leaf = None;
                        }
                        return None;
                    }
                }
            }
        }
    }
}
impl<'a, K, V, const N: usize, const M: usize> DoubleEndedIterator for Range<'a, K, V, N, M> {
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(f) = &mut self.bck_leaf {
                if let Some(x) = f.next_back() {
                    return Some(x);
                }
                self.bck_leaf = None;
            } else if let Some(s) = self.bck_stk.last_mut() {
                if let Some(kv) = s.v.next_back() {
                    if let Some(ct) = s.c.next_back() {
                        self.push_tree_back(ct);
                    }
                    return Some(kv);
                }
                self.bck_stk.pop();
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
                            }
                            self.fwd_leaf = None;
                        }
                        return None;
                    }
                }
            }
        }
    }
}
impl<'a, K, V, const N: usize, const M: usize> FusedIterator for Range<'a, K, V, N, M> {}

//////////////////////////////////////////// END RANGE ////////////////////////////

//#[derive(Clone, Debug)]
struct StkMut<'a, K, V, const N: usize, const M: usize> {
    v: LeafIterMut<'a, K, V, N>,
    c: ChildIterMut<'a, K, V, N, M>,
}

/// Iterator returned by [`BTreeMap::range_mut`].
//#[derive(Clone, Debug, Default)]
pub struct RangeMut<'a, K, V, const N: usize, const M: usize> {
    fwd_leaf: Option<LeafIterMut<'a, K, V, N>>,
    bck_leaf: Option<LeafIterMut<'a, K, V, N>>,
    fwd_stk: StkVec<StkMut<'a, K, V, N, M>>,
    bck_stk: StkVec<StkMut<'a, K, V, N, M>>,
}
impl<'a, K, V, const N: usize, const M: usize> RangeMut<'a, K, V, N, M> {
    fn new() -> Self {
        Self {
            fwd_leaf: None,
            bck_leaf: None,
            fwd_stk: StkVec::new(),
            bck_stk: StkVec::new(),
        }
    }

    fn push_tree(&mut self, tree: TreeRefMut<'a, K, V, N, M>, both: bool) {
        match tree {
            TreeRefMut::L(leaf, len) => {
                self.fwd_leaf = Some(leaf.iter_mut(len));
            }
            TreeRefMut::NL(nl, len) => {
                let (v, mut c) = nl.iter_mut(len);
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

    fn push_tree_back(&mut self, tree: TreeRefMut<'a, K, V, N, M>) {
        match tree {
            TreeRefMut::L(leaf, len) => {
                self.bck_leaf = Some(leaf.iter_mut(len));
            }
            TreeRefMut::NL(nl, len) => {
                let (v, mut c) = nl.iter_mut(len);
                let ct_back = c.next_back();
                self.bck_stk.push(StkMut { v, c });
                if let Some(ct_back) = ct_back {
                    self.push_tree_back(ct_back);
                }
            }
        }
    }
    fn steal_bck(&mut self) -> StealResultMut<'a, K, V, N, M> {
        for s in &mut self.bck_stk {
            if s.v.len() > s.c.len() {
                return StealResultMut::KV(s.v.next().unwrap());
            } else if let Some(ct) = s.c.next() {
                return StealResultMut::CT(ct);
            }
        }
        StealResultMut::Nothing
    }
    fn steal_fwd(&mut self) -> StealResultMut<'a, K, V, N, M> {
        for s in &mut self.fwd_stk {
            if s.v.len() > s.c.len() {
                return StealResultMut::KV(s.v.next_back().unwrap());
            } else if let Some(ct) = s.c.next_back() {
                return StealResultMut::CT(ct);
            }
        }
        StealResultMut::Nothing
    }
}
impl<'a, K, V, const N: usize, const M: usize> Iterator for RangeMut<'a, K, V, N, M> {
    type Item = (&'a K, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(f) = &mut self.fwd_leaf {
                if let Some(x) = f.next() {
                    return Some(x);
                }
                self.fwd_leaf = None;
            } else if let Some(s) = self.fwd_stk.last_mut() {
                if let Some(kv) = s.v.next() {
                    if let Some(ct) = s.c.next() {
                        self.push_tree(ct, false);
                    }
                    return Some(kv);
                }
                self.fwd_stk.pop();
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
                            }
                            self.bck_leaf = None;
                        }
                        return None;
                    }
                }
            }
        }
    }
}
impl<'a, K, V, const N: usize, const M: usize> DoubleEndedIterator for RangeMut<'a, K, V, N, M> {
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(f) = &mut self.bck_leaf {
                if let Some(x) = f.next_back() {
                    return Some(x);
                }
                self.bck_leaf = None;
            } else if let Some(s) = self.bck_stk.last_mut() {
                if let Some(kv) = s.v.next_back() {
                    if let Some(ct) = s.c.next_back() {
                        self.push_tree_back(ct);
                    }
                    return Some(kv);
                }
                self.bck_stk.pop();
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
                            }
                            self.fwd_leaf = None;
                        }
                        return None;
                    }
                }
            }
        }
    }
}
impl<'a, K, V, const N: usize, const M: usize> FusedIterator for RangeMut<'a, K, V, N, M> {}

enum StealResult<'a, K, V, const N: usize, const M: usize> {
    KV((&'a K, &'a V)),          // Key-value pair.
    CT(TreeRef<'a, K, V, N, M>), // Child Tree.
    Nothing,
}

enum StealResultMut<'a, K, V, const N: usize, const M: usize> {
    KV((&'a K, &'a mut V)),         // Key-value pair.
    CT(TreeRefMut<'a, K, V, N, M>), // Child Tree.
    Nothing,
}

/// Iterator returned by [`BTreeMap::iter`].
// #[derive(Clone, Debug, Default)]
pub struct Iter<'a, K, V, const N: usize, const M: usize> {
    len: usize,
    inner: Range<'a, K, V, N, M>,
}
impl<'a, K, V, const N: usize, const M: usize> Iterator for Iter<'a, K, V, N, M> {
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
impl<'a, K, V, const N: usize, const M: usize> ExactSizeIterator for Iter<'a, K, V, N, M> {
    fn len(&self) -> usize {
        self.len
    }
}
impl<'a, K, V, const N: usize, const M: usize> DoubleEndedIterator for Iter<'a, K, V, N, M> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            self.inner.next_back()
        }
    }
}
impl<'a, K, V, const N: usize, const M: usize> FusedIterator for Iter<'a, K, V, N, M> {}

/// Iterator returned by [`BTreeMap::iter_mut`].
// #[derive(Clone, Debug, Default)]
pub struct IterMut<'a, K, V, const N: usize, const M: usize> {
    len: usize,
    inner: RangeMut<'a, K, V, N, M>,
}
impl<'a, K, V, const N: usize, const M: usize> Iterator for IterMut<'a, K, V, N, M> {
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
impl<'a, K, V, const N: usize, const M: usize> ExactSizeIterator for IterMut<'a, K, V, N, M> {
    fn len(&self) -> usize {
        self.len
    }
}
impl<'a, K, V, const N: usize, const M: usize> DoubleEndedIterator for IterMut<'a, K, V, N, M> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            self.inner.next_back()
        }
    }
}
impl<'a, K, V, const N: usize, const M: usize> FusedIterator for IterMut<'a, K, V, N, M> {}

// NonLeaf ...

enum CA<K, V, const N: usize, const M: usize> {
    L(MaybeUninit<[Box<Leaf<K, V, N>>; M]>),
    NL(MaybeUninit<[Box<NonLeaf<K, V, N, M>>; M]>),
}

struct NonLeaf<K, V, const N: usize, const M: usize> {
    leaf: Leaf<K, V, N>,
    c: CA<K, V, N, M>,
    clen: [u8; M],
}

impl<K, V, const N: usize, const M: usize> NonLeaf<K, V, N, M> {
    fn iter(&self, len: &u8) -> (LeafIter<K, V, N>, ChildIter<K, V, N, M>) {
        let n = *len as usize + 1;
        let ci = unsafe {
            let len = self.clen[0..n].iter();
            match &self.c {
                CA::L(a) => ChildIter::L((*a.as_ptr())[0..n].iter(), len),
                CA::NL(a) => ChildIter::NL((*a.as_ptr())[0..n].iter(), len),
            }
        };
        (self.leaf.iter(len), ci)
    }

    fn iter_mut(&mut self, len: &u8) -> (LeafIterMut<K, V, N>, ChildIterMut<K, V, N, M>) {
        let n = *len as usize + 1;
        let cim = unsafe {
            let len = self.clen[0..n].iter();
            match &mut self.c {
                CA::L(a) => ChildIterMut::L((*a.as_mut_ptr())[0..n].iter_mut(), len),
                CA::NL(a) => ChildIterMut::NL((*a.as_mut_ptr())[0..n].iter_mut(), len),
            }
        };
        (self.leaf.iter_mut(len), cim)
    }

    fn print_clen(&self, n: usize) {
        println!("clen({})={:?}", n + 1, &self.clen[0..n + 1]);
    }

    fn new(child_is_leaf: bool) -> Self {
        Self {
            leaf: Leaf::new(),
            c: if child_is_leaf {
                CA::L(MaybeUninit::uninit())
            } else {
                CA::NL(MaybeUninit::uninit())
            },
            clen: [0; M],
        }
    }

    /// Cannot implement Drop as len needs to be a parameter.
    /// Instead this function is called to drop any stored key-value pairs and child nodes.
    fn free(&mut self, len: &mut u8) {
        let n = *len as usize;
        self.leaf.free(len);
        for i in 0..=n {
            self.child_free(i);
        }
    }

    fn get<Q>(&self, len: usize, key: &Q) -> Option<&V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self.leaf.search(len, |x| x.borrow().cmp(key)) {
            Ok(i) => Some(self.leaf.ixv(i)),
            Err(i) => self.child_get(i, key),
        }
    }

    fn get_key_value<Q>(&self, len: usize, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self.leaf.search(len, |x| x.borrow().cmp(key)) {
            Ok(i) => Some(self.leaf.kv(i)),
            Err(i) => self.child_get_key_value(i, key),
        }
    }

    fn get_mut<Q>(&mut self, len: usize, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self.leaf.search(len, |x| x.borrow().cmp(key)) {
            Ok(i) => Some(self.leaf.ixvm(i)),
            Err(i) => self.child_get_mut(i, key),
        }
    }

    fn insert_kv(&mut self, len: &mut u8, key: K, x: &mut InsertCtx<K, V, N, M>)
    where
        K: Ord,
    {
        let n = *len as usize;
        match self.leaf.search(n, |k| k.borrow().cmp(&key)) {
            Ok(i) => {
                let value = x.value.take().unwrap();
                x.value = Some(std::mem::replace(self.leaf.ixvm(i), value));
                *self.leaf.ixkm(i) = key;
            }
            Err(i) => {
                self.child_insert_kv(i, key, x);
                if let Some(((k, v), right, rlen)) = x.split.take() {
                    self.insert_child(n + 1, i + 1, rlen, right);
                    self.leaf.insert(n, i, k, v);
                    *len += 1;
                    // This should really be delayed until next insert, but this is simpler for now.
                    if *len as usize == N {
                        x.split = Some(self.split(len));
                    }
                }
            }
        }
    }

    fn pop_first(&mut self, len: &mut u8) -> Option<(K, V)> {
        if let Some(x) = self.child_pop_first(0) {
            Some(x)
        } else if *len == 0 {
            None
        } else {
            self.child_free(0);
            self.leaf.pop_first(len)
        }
    }

    fn pop_last(&mut self, len: &mut u8) -> Option<(K, V)> {
        let i = *len as usize;
        if let Some(x) = self.child_pop_last(i) {
            Some(x)
        } else if i == 0 {
            None
        } else {
            self.child_free(i);
            self.leaf.pop_last(len)
        }
    }

    fn remove_entry<Q>(&mut self, len: &mut u8, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self.leaf.search(*len as usize, |x| x.borrow().cmp(key)) {
            Ok(i) => Some(self.remove_at(len, i).0),
            Err(i) => self.child_remove(i, key),
        }
    }

    fn child_is_leaf(&self) -> bool {
        match self.c {
            CA::L(_) => true,
            CA::NL(_) => false,
        }
    }

    // Split non-leaf
    fn split(&mut self, len: &mut u8) -> Split<K, V, N, M> {
        assert!(*len as usize == N);
        let mut right = Self::new(self.child_is_leaf());
        self.leaf.mov(N / 2 + 1, N / 2, &mut right.leaf);
        *len = (N / 2 + 1) as u8;
        let med = self.leaf.pop_last(len).unwrap();

        let at = M / 2;
        let n = M / 2;
        unsafe {
            match &mut self.c {
                CA::L(a) => {
                    let from = ix(a.as_ptr(), at);
                    match &mut right.c {
                        CA::L(a) => {
                            let to = ixm(a.as_mut_ptr(), 0);
                            ptr::copy_nonoverlapping(from, to, n);
                        }
                        CA::NL(_) => panic!(),
                    }
                }
                CA::NL(a) => {
                    let from = ix(a.as_ptr(), at);
                    match &mut right.c {
                        CA::NL(a) => {
                            let to = ixm(a.as_mut_ptr(), 0);
                            ptr::copy_nonoverlapping(from, to, n);
                        }
                        CA::L(_) => panic!(),
                    }
                }
            }
        }
        for i in 0..n {
            right.clen[i] = self.clen[at + i]
        }
        (med, Tree::NL(Box::new(right)), N / 2)
    }

    fn child_free(&mut self, at: usize) {
        let clen = &mut self.clen[at];
        unsafe {
            match &mut self.c {
                CA::L(a) => {
                    let p = ixm(a.as_mut_ptr(), at);
                    (*p).free(clen);
                    let _ = p.read();
                }
                CA::NL(a) => {
                    let p = ixm(a.as_mut_ptr(), at);
                    (*p).free(clen);
                    let _ = p.read();
                }
            }
        }
    }

    fn child_pop_first(&mut self, at: usize) -> Option<(K, V)> {
        let clen = &mut self.clen[at];
        unsafe {
            match &mut self.c {
                CA::L(a) => (*ixm(a.as_mut_ptr(), at)).pop_first(clen),
                CA::NL(a) => (*ixm(a.as_mut_ptr(), at)).pop_first(clen),
            }
        }
    }

    fn child_pop_last(&mut self, at: usize) -> Option<(K, V)> {
        let clen = &mut self.clen[at];
        unsafe {
            match &mut self.c {
                CA::L(a) => (*ixm(a.as_mut_ptr(), at)).pop_last(clen),
                CA::NL(a) => (*ixm(a.as_mut_ptr(), at)).pop_last(clen),
            }
        }
    }

    fn child_get<Q>(&self, at: usize, key: &Q) -> Option<&V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let clen = self.clen[at] as usize;
        unsafe {
            match &self.c {
                CA::L(a) => (*ix(a.as_ptr(), at)).get(clen, key),
                CA::NL(a) => (*ix(a.as_ptr(), at)).get(clen, key),
            }
        }
    }

    fn child_get_key_value<Q>(&self, at: usize, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let clen = self.clen[at] as usize;
        unsafe {
            match &self.c {
                CA::L(a) => (*ix(a.as_ptr(), at)).get_key_value(clen, key),
                CA::NL(a) => (*ix(a.as_ptr(), at)).get_key_value(clen, key),
            }
        }
    }

    fn child_get_mut<Q>(&mut self, at: usize, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let clen = self.clen[at] as usize;
        unsafe {
            match &mut self.c {
                CA::L(a) => (*ixm(a.as_mut_ptr(), at)).get_mut(clen, key),
                CA::NL(a) => (*ixm(a.as_mut_ptr(), at)).get_mut(clen, key),
            }
        }
    }

    fn child_insert_kv(&mut self, at: usize, key: K, x: &mut InsertCtx<K, V, N, M>)
    where
        K: Ord,
    {
        let clen = &mut self.clen[at];
        unsafe {
            match &mut self.c {
                CA::L(a) => (*ixm(a.as_mut_ptr(), at)).insert_kv(clen, key, x),
                CA::NL(a) => (*ixm(a.as_mut_ptr(), at)).insert_kv(clen, key, x),
            }
        }
    }

    fn child_remove<Q>(&mut self, at: usize, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let clen = &mut self.clen[at];
        unsafe {
            match &mut self.c {
                CA::L(a) => (*ixm(a.as_mut_ptr(), at)).remove_entry(clen, key),
                CA::NL(a) => (*ixm(a.as_mut_ptr(), at)).remove_entry(clen, key),
            }
        }
    }

    fn remove_at(&mut self, len: &mut u8, i: usize) -> ((K, V), usize) {
        if let Some(x) = self.child_pop_last(i) {
            (self.leaf.replace(i, x), 0)
        } else {
            self.remove_child(*len as usize + 1, i);
            (self.leaf.remove(len, i), 1)
        }
    }

    fn push_child(&mut self, len: usize, child: Tree<K, V, N, M>) {
        assert!(len < M);
        unsafe {
            match &mut self.c {
                CA::L(a) => {
                    let p = ixm(a.as_mut_ptr(), len);
                    p.write(child.leaf());
                }
                CA::NL(a) => {
                    let p = ixm(a.as_mut_ptr(), len);
                    p.write(child.non_leaf());
                }
            }
        }
    }

    fn insert_child(&mut self, len: usize, at: usize, clen: usize, child: Tree<K, V, N, M>) {
        unsafe {
            let n = len - at;
            match &mut self.c {
                CA::L(a) => {
                    let p = ixm(a.as_mut_ptr(), at);
                    if n > 0 {
                        ptr::copy(p, p.add(1), n);
                    }
                    p.write(child.leaf());
                }
                CA::NL(a) => {
                    let p = ixm(a.as_mut_ptr(), at);
                    if n > 0 {
                        ptr::copy(p, p.add(1), n);
                    }
                    p.write(child.non_leaf());
                }
            }
        }
        let mut i = len;
        while i > at {
            self.clen[i] = self.clen[i - 1];
            i -= 1;
        }
        self.clen[at] = clen as u8;
    }

    fn remove_child(&mut self, len: usize, at: usize) {
        self.child_free(at);
        unsafe {
            let n = len - at - 1;
            match &mut self.c {
                CA::L(a) => {
                    let p = ixm(a.as_mut_ptr(), at);
                    if n > 0 {
                        ptr::copy(p.add(1), p, n);
                    }
                }
                CA::NL(a) => {
                    let p = ixm(a.as_mut_ptr(), at);
                    if n > 0 {
                        ptr::copy(p.add(1), p, n);
                    }
                }
            }
            for i in at..at + n {
                self.clen[i] = self.clen[i + 1];
            }
        }
    }
} // end impl NonLeaf

/// Cursor that allows mutation of map, returned by [`BTreeMap::lower_bound_mut`], [`BTreeMap::upper_bound_mut`].
pub struct CursorMut<'a, K, V, const N: usize, const M: usize>(CursorMutKey<'a, K, V, N, M>);
impl<'a, K, V, const N: usize, const M: usize> CursorMut<'a, K, V, N, M> {
    fn lower_bound<Q>(map: &'a mut BTreeMap<K, V, N, M>, bound: Bound<&Q>) -> Self
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        unsafe {
            // Converting map to raw pointer here is necessary to keep Miri happy
            // although not when using MIRIFLAGS=-Zmiri-tree-borrows.
            let clen = map.clen as usize;
            let map: *mut BTreeMap<K, V, N, M> = map;
            let mut s = CursorMutKey::make(map);
            s.push_lower(clen, &mut (*map).tree, bound);
            Self(s)
        }
    }

    fn upper_bound<Q>(map: &'a mut BTreeMap<K, V, N, M>, bound: Bound<&Q>) -> Self
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        unsafe {
            // Converting map to raw pointer here is necessary to keep Miri happy
            // although not when using MIRIFLAGS=-Zmiri-tree-borrows.
            let clen = map.clen as usize;
            let map: *mut BTreeMap<K, V, N, M> = map;
            let mut s = CursorMutKey::make(map);
            s.push_upper(clen, &mut (*map).tree, bound);
            Self(s)
        }
    }

    /// Advance the cursor, returns references to the key and value of the element that it moved over.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<(&K, &mut V)>
    where
        K: Ord,
    {
        let (k, v) = self.0.next()?;
        Some((&*k, v))
    }

    /// Move the cursor back, returns references to the key and value of the element that it moved over.
    pub fn prev(&mut self) -> Option<(&K, &mut V)>
    where
        K: Ord,
    {
        let (k, v) = self.0.prev()?;
        Some((&*k, v))
    }

    /// Get references to the next key/value pair.
    pub fn peek_next(&self) -> Option<(&K, &mut V)> {
        let (k, v) = self.0.peek_next()?;
        Some((&*k, v))
    }

    /// Get references to the previous key/value pair.
    pub fn peek_prev(&self) -> Option<(&K, &mut V)> {
        let (k, v) = self.0.peek_prev()?;
        Some((&*k, v))
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

    /// Converts the cursor into a `CursorMutKey`, which allows mutating the key of elements in the tree.
    /// # Safety
    ///
    /// Keys must be unique and in sorted order.
    #[must_use]
    pub unsafe fn with_mutable_key(self) -> CursorMutKey<'a, K, V, N, M> {
        self.0
    }

    /// Returns a read-only cursor pointing to the same location as the `CursorMut`.
    #[must_use]
    pub fn as_cursor(&self) -> Cursor<'_, K, V, N, M> {
        self.0.as_cursor()
    }

    /// This is needed for the implementation of the [Entry] API.
    fn into_mut(self) -> &'a mut V {
        self.0.into_mut()
    }
}

type StkVec<T> = arrayvec::ArrayVec<T, 15>;

/// Cursor that allows mutation of map keys, returned by [`CursorMut::with_mutable_key`].
pub struct CursorMutKey<'a, K, V, const N: usize, const M: usize> {
    map: *mut BTreeMap<K, V, N, M>,
    leaf: Option<*mut Leaf<K, V, N>>,
    index: usize,
    len: usize,
    stack: StkVec<(*mut NonLeaf<K, V, N, M>, usize, usize)>,
    _pd: PhantomData<&'a mut BTreeMap<K, V, N, M>>,
}

unsafe impl<'a, K, V, const N: usize, const M: usize> Send for CursorMutKey<'a, K, V, N, M> {}
unsafe impl<'a, K, V, const N: usize, const M: usize> Sync for CursorMutKey<'a, K, V, N, M> {}

impl<'a, K, V, const N: usize, const M: usize> CursorMutKey<'a, K, V, N, M> {
    fn make(map: *mut BTreeMap<K, V, N, M>) -> Self {
        Self {
            map,
            leaf: None,
            index: 0,
            len: 0,
            stack: StkVec::new(),
            _pd: PhantomData,
        }
    }

    fn push_lower<Q>(&mut self, len: usize, tree: &mut Tree<K, V, N, M>, bound: Bound<&Q>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match tree {
            Tree::L(leaf) => self.push_lower_leaf(len, leaf, bound),
            Tree::NL(nl) => self.push_lower_nl(len, nl, bound),
        }
    }

    fn push_upper<Q>(&mut self, len: usize, tree: &mut Tree<K, V, N, M>, bound: Bound<&Q>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match tree {
            Tree::L(leaf) => self.push_upper_leaf(len, leaf, bound),
            Tree::NL(nl) => self.push_upper_nl(len, nl, bound),
        }
    }

    fn push_lower_leaf<Q>(&mut self, len: usize, leaf: &mut Leaf<K, V, N>, bound: Bound<&Q>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.index = leaf.get_lower(len, bound);
        self.len = len;
        self.leaf = Some(leaf);
    }

    fn push_upper_leaf<Q>(&mut self, len: usize, leaf: &mut Leaf<K, V, N>, bound: Bound<&Q>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.index = leaf.get_upper(len, bound);
        self.len = len;
        self.leaf = Some(leaf);
    }

    fn push_lower_nl<Q>(&mut self, len: usize, nl: &mut NonLeaf<K, V, N, M>, bound: Bound<&Q>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        unsafe {
            let ix = nl.leaf.get_lower(len, bound);
            self.stack.push((nl, ix, len));
            let len = nl.clen[ix] as usize;
            match &mut nl.c {
                CA::L(a) => {
                    let p = ixm(a.as_mut_ptr(), ix);
                    self.push_lower_leaf(len, &mut *p, bound);
                }
                CA::NL(a) => {
                    let p = ixm(a.as_mut_ptr(), ix);
                    self.push_lower_nl(len, &mut *p, bound);
                }
            }
        }
    }

    fn push_upper_nl<Q>(&mut self, len: usize, nl: &mut NonLeaf<K, V, N, M>, bound: Bound<&Q>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        unsafe {
            let ix = nl.leaf.get_upper(len, bound);
            self.stack.push((nl, ix, len));
            let len = nl.clen[ix] as usize;
            match &mut nl.c {
                CA::L(a) => {
                    let p = ixm(a.as_mut_ptr(), ix);
                    self.push_upper_leaf(len, &mut *p, bound);
                }
                CA::NL(a) => {
                    let p = ixm(a.as_mut_ptr(), ix);
                    self.push_upper_nl(len, &mut *p, bound);
                }
            }
        }
    }

    fn push_child(&mut self, tsp: usize, nl: *mut NonLeaf<K, V, N, M>, ix: usize, len: usize) {
        self.stack[tsp] = (nl, ix, len);
        unsafe {
            let len = (*nl).clen[ix] as usize;
            match &mut (*nl).c {
                CA::L(a) => {
                    self.index = 0;
                    self.len = len;
                    let p = ixm(a.as_mut_ptr(), ix);
                    self.leaf = Some(&mut **p);
                }
                CA::NL(a) => {
                    let p = ixm(a.as_mut_ptr(), ix);
                    self.push_child(tsp + 1, &mut **p, 0, len);
                }
            }
        }
    }

    fn push_child_back(&mut self, tsp: usize, nl: *mut NonLeaf<K, V, N, M>, ix: usize, len: usize) {
        self.stack[tsp] = (nl, ix, len);
        unsafe {
            let len = (*nl).clen[ix] as usize;
            match &mut (*nl).c {
                CA::L(a) => {
                    self.index = len;
                    self.len = len;
                    let p = ixm(a.as_mut_ptr(), ix);
                    self.leaf = Some(&mut **p);
                }
                CA::NL(a) => {
                    let p = ixm(a.as_mut_ptr(), ix);
                    self.push_child_back(tsp + 1, &mut **p, len, len);
                }
            }
        }
    }

    /// Advance the cursor, returns references to the key and value of the element that it moved over.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<(&mut K, &mut V)> {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            if self.index < self.len {
                self.index += 1;
                Some((*leaf).kvm(self.index - 1))
            } else {
                let mut tsp = self.stack.len();
                while tsp > 0 {
                    tsp -= 1;
                    let (nl, ix, len) = self.stack[tsp];
                    if ix < len {
                        self.push_child(tsp, nl, ix + 1, len);
                        return Some((*nl).leaf.kvm(ix));
                    }
                }
                None
            }
        }
    }

    /// Remove next element.
    pub fn remove_next(&mut self) -> Option<(K, V)> {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            if self.index < self.len {
                (*self.map).len -= 1;
                self.len -= 1;
                let lenp = self.get_lenp();
                Some((*leaf).remove(&mut *lenp, self.index))
            } else {
                let mut tsp = self.stack.len();
                while tsp > 0 {
                    tsp -= 1;
                    let (nl, mut ix, mut len) = self.stack[tsp];
                    if ix < len {
                        (*self.map).len -= 1;
                        let lenp = self.get_lenp2(tsp);
                        let (kv, removed) = (*nl).remove_at(&mut *lenp, ix);
                        ix += 1 - removed;
                        len -= removed;
                        self.push_child(tsp, nl, ix, len);
                        return Some(kv);
                    }
                }
                None
            }
        }
    }

    /// Remove previous element.
    pub fn remove_prev(&mut self) -> Option<(K, V)> {
        self.prev()?;
        self.remove_next()
    }

    /// Move the cursor back, returns references to the key and value of the element that it moved over.
    pub fn prev(&mut self) -> Option<(&mut K, &mut V)> {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            if self.index > 0 {
                self.index -= 1;
                Some((*leaf).kvm(self.index))
            } else {
                let mut tsp = self.stack.len();
                while tsp > 0 {
                    tsp -= 1;
                    let (nl, ix, len) = self.stack[tsp];
                    if ix > 0 {
                        self.push_child_back(tsp, nl, ix - 1, len);
                        return Some((*nl).leaf.kvm(ix - 1));
                    }
                }
                None
            }
        }
    }

    /// Get references to the next key and value.
    pub fn peek_next(&self) -> Option<(&mut K, &mut V)> {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            if self.index < self.len {
                Some((*leaf).kvm(self.index))
            } else {
                let mut tsp = self.stack.len();
                while tsp > 0 {
                    tsp -= 1;
                    let (nl, ix, len) = self.stack[tsp];
                    if ix < len {
                        return Some((*nl).leaf.kvm(ix));
                    }
                }
                None
            }
        }
    }

    /// Get references to the previous key and value.
    pub fn peek_prev(&self) -> Option<(&mut K, &mut V)> {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            if self.index > 0 {
                Some((*leaf).kvm(self.index - 1))
            } else {
                let mut tsp = self.stack.len();
                while tsp > 0 {
                    tsp -= 1;
                    let (nl, ix, _len) = self.stack[tsp];
                    if ix > 0 {
                        return Some((*nl).leaf.kvm(ix - 1));
                    }
                }
                None
            }
        }
    }

    unsafe fn get_lenp(&mut self) -> *mut u8 {
        let sp = self.stack.len();
        if sp == 0 {
            &mut (*self.map).clen
        } else {
            let (nl, ix, _len) = self.stack[sp - 1];
            &mut (*nl).clen[ix]
        }
    }

    unsafe fn get_lenp2(&mut self, sp: usize) -> *mut u8 {
        if sp == 0 {
            &mut (*self.map).clen
        } else {
            let (nl, ix, _len) = self.stack[sp - 1];
            &mut (*nl).clen[ix]
        }
    }

    /// After split, we need to re-calculate the leaf from the parent.
    unsafe fn get_leaf(&mut self) -> *mut Leaf<K, V, N> {
        let (nl, index, _len) = self.stack[self.stack.len() - 1];
        match &mut (*nl).c {
            CA::L(a) => &mut **ixm(a.as_mut_ptr(), index),
            CA::NL(_) => panic!(),
        }
    }

    /// After split, we need to re-calculate from the parent.
    unsafe fn get_nonleaf(&mut self) -> *mut NonLeaf<K, V, N, M> {
        let (nl, index, _len) = self.stack[self.stack.len() - 1];
        match &mut (*nl).c {
            CA::NL(a) => &mut **ixm(a.as_mut_ptr(), index),
            CA::L(_) => panic!(),
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
            if self.index == N {
                // Leaf is full.
                let lenp = self.get_lenp();
                let (med, right) = (*leaf).split(&mut *lenp);

                let right = Tree::L(right);
                let r = usize::from(self.index > N / 2);
                self.index -= r * (N / 2 + 1);
                self.save_split(med, right, r);
                self.len = N / 2;
                leaf = self.get_leaf();
                self.leaf = Some(leaf);
            }
            (*leaf).insert(self.len, self.index, key, value);
            (*self.get_lenp()) += 1;
            self.len += 1;
        }
    }

    // This is called when a child has to split and there is a tree to be saved in the parent.
    fn save_split(&mut self, med: (K, V), tree: Tree<K, V, N, M>, r: usize) {
        let rlen = N / 2;
        unsafe {
            if let Some((mut nl, mut ix, mut len)) = self.stack.pop() {
                if len == N {
                    let lenp = self.get_lenp();
                    assert_eq!(len, *self.get_lenp() as usize);
                    let (med, tree, _) = (*nl).split(&mut *lenp);
                    let r = usize::from(ix > N / 2);
                    ix -= r * (N / 2 + 1);
                    self.save_split(med, tree, r);
                    nl = self.get_nonleaf();
                    len = *self.get_lenp() as usize;
                }
                (*nl).leaf.insert(len, ix, med.0, med.1);
                (*nl).insert_child(len + 1, ix + 1, rlen, tree);
                ix += r;
                assert!(len == *self.get_lenp() as usize);
                *self.get_lenp() += 1;
                len += 1;
                self.stack.push((nl, ix, len));
            } else {
                (*self.map).new_root((med, tree, rlen));
                let nl = (*self.map).tree.non_leaf_ptr();
                self.stack.push((nl, r, 1));
            }
        }
    }

    /// Returns a read-only cursor pointing to the same location as the `CursorMutKey`.
    #[must_use]
    pub fn as_cursor(&self) -> Cursor<'_, K, V, N, M> {
        unsafe {
            let mut c = Cursor::make();
            c.index = self.index;
            c.leaf = Some(&*self.leaf.unwrap());
            for (nl, ix, len) in &self.stack {
                c.stack.push((&(**nl), *ix, *len));
            }
            c
        }
    }

    /// This is needed for the implementation of the [Entry] API.
    fn into_mut(self) -> &'a mut V {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            (*leaf).ixvm(self.index)
        }
    }
}

/// Cursor returned by [`BTreeMap::lower_bound`], [`BTreeMap::upper_bound`].
#[derive(Debug, Clone)]
pub struct Cursor<'a, K, V, const N: usize, const M: usize> {
    leaf: Option<*const Leaf<K, V, N>>,
    index: usize,
    len: usize,
    stack: StkVec<(*const NonLeaf<K, V, N, M>, usize, usize)>,
    _pd: PhantomData<&'a BTreeMap<K, V, N, M>>,
}

unsafe impl<'a, K, V, const N: usize, const M: usize> Send for Cursor<'a, K, V, N, M> {}
unsafe impl<'a, K, V, const N: usize, const M: usize> Sync for Cursor<'a, K, V, N, M> {}

impl<'a, K, V, const N: usize, const M: usize> Cursor<'a, K, V, N, M> {
    fn make() -> Self {
        Self {
            leaf: None,
            index: 0,
            len: 0,
            stack: StkVec::new(),
            _pd: PhantomData,
        }
    }

    fn lower_bound<Q>(map: &'a BTreeMap<K, V, N, M>, bound: Bound<&Q>) -> Self
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let clen = map.clen as usize;
        let mut s = Self::make();
        s.push_lower(clen, &map.tree, bound);
        s
    }

    fn upper_bound<Q>(map: &'a BTreeMap<K, V, N, M>, bound: Bound<&Q>) -> Self
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let clen = map.clen as usize;
        let mut s = Self::make();
        s.push_upper(clen, &map.tree, bound);
        s
    }

    fn push_lower<Q>(&mut self, len: usize, tree: &Tree<K, V, N, M>, bound: Bound<&Q>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match tree {
            Tree::L(leaf) => self.push_lower_leaf(len, leaf, bound),
            Tree::NL(nl) => self.push_lower_nl(len, nl, bound),
        }
    }

    fn push_upper<Q>(&mut self, len: usize, tree: &Tree<K, V, N, M>, bound: Bound<&Q>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match tree {
            Tree::L(leaf) => self.push_upper_leaf(len, leaf, bound),
            Tree::NL(nl) => self.push_upper_nl(len, nl, bound),
        }
    }

    fn push_lower_leaf<Q>(&mut self, len: usize, leaf: &Leaf<K, V, N>, bound: Bound<&Q>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.index = leaf.get_lower(len, bound);
        self.len = len;
        self.leaf = Some(leaf);
    }

    fn push_upper_leaf<Q>(&mut self, len: usize, leaf: &Leaf<K, V, N>, bound: Bound<&Q>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.index = leaf.get_upper(len, bound);
        self.len = len;
        self.leaf = Some(leaf);
    }

    fn push_lower_nl<Q>(&mut self, len: usize, nl: &NonLeaf<K, V, N, M>, bound: Bound<&Q>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        unsafe {
            let i = nl.leaf.get_lower(len, bound);
            self.stack.push((nl, i, len));
            let len = nl.clen[i] as usize;
            match &nl.c {
                CA::L(a) => {
                    let p = ix(a.as_ptr(), i);
                    self.push_lower_leaf(len, &*p, bound);
                }
                CA::NL(a) => {
                    let p = ix(a.as_ptr(), i);
                    self.push_lower_nl(len, &*p, bound);
                }
            }
        }
    }

    fn push_upper_nl<Q>(&mut self, len: usize, nl: &NonLeaf<K, V, N, M>, bound: Bound<&Q>)
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        unsafe {
            let i = nl.leaf.get_upper(len, bound);
            self.stack.push((nl, i, len));
            let len = nl.clen[i] as usize;
            match &nl.c {
                CA::L(a) => {
                    let p = ix(a.as_ptr(), i);
                    self.push_upper_leaf(len, &*p, bound);
                }
                CA::NL(a) => {
                    let p = ix(a.as_ptr(), i);
                    self.push_upper_nl(len, &*p, bound);
                }
            }
        }
    }

    fn push_child(&mut self, tsp: usize, nl: *const NonLeaf<K, V, N, M>, i: usize, len: usize) {
        self.stack[tsp] = (nl, i, len);
        unsafe {
            let len = (*nl).clen[i] as usize;
            match &(*nl).c {
                CA::L(a) => {
                    self.index = 0;
                    self.len = len;
                    let p = ix(a.as_ptr(), i);
                    self.leaf = Some(&**p);
                }
                CA::NL(a) => {
                    let p = ix(a.as_ptr(), i);
                    self.push_child(tsp + 1, &**p, 0, len);
                }
            }
        }
    }

    fn push_child_back(
        &mut self,
        tsp: usize,
        nl: *const NonLeaf<K, V, N, M>,
        i: usize,
        len: usize,
    ) {
        self.stack[tsp] = (nl, i, len);
        unsafe {
            let len = (*nl).clen[i] as usize;
            match &(*nl).c {
                CA::L(a) => {
                    self.index = len;
                    self.len = len;
                    let p = ix(a.as_ptr(), i);
                    self.leaf = Some(&**p);
                }
                CA::NL(a) => {
                    let p = ix(a.as_ptr(), i);
                    self.push_child_back(tsp + 1, &**p, len, len);
                }
            }
        }
    }

    /// Advance the cursor, returns references to the key and value of the element that it moved over.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<(&K, &V)> {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            if self.index < self.len {
                self.index += 1;
                Some((*leaf).kv(self.index - 1))
            } else {
                let mut tsp = self.stack.len();
                while tsp > 0 {
                    tsp -= 1;
                    let (nl, ix, len) = self.stack[tsp];
                    if ix < len {
                        self.push_child(tsp, nl, ix + 1, len);
                        return Some((*nl).leaf.kv(ix));
                    }
                }
                None
            }
        }
    }

    /// Move the cursor back, returns references to the key and value of the element that it moved over.
    pub fn prev(&mut self) -> Option<(&K, &V)> {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            if self.index > 0 {
                self.index -= 1;
                Some((*leaf).kv(self.index))
            } else {
                let mut tsp = self.stack.len();
                while tsp > 0 {
                    tsp -= 1;
                    let (nl, ix, len) = self.stack[tsp];
                    if ix > 0 {
                        self.push_child_back(tsp, nl, ix - 1, len);
                        return Some((*nl).leaf.kv(ix - 1));
                    }
                }
                None
            }
        }
    }

    /// Get references to the next key and value.
    pub fn peek_next(&self) -> Option<(&K, &V)> {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            if self.index < self.len {
                Some((*leaf).kv(self.index))
            } else {
                let mut tsp = self.stack.len();
                while tsp > 0 {
                    tsp -= 1;
                    let (nl, ix, len) = self.stack[tsp];
                    if ix < len {
                        return Some((*nl).leaf.kv(ix));
                    }
                }
                None
            }
        }
    }

    /// Get references to the previous key and value.
    pub fn peek_prev(&self) -> Option<(&K, &V)> {
        unsafe {
            let leaf = self.leaf.unwrap_unchecked();
            if self.index > 0 {
                Some((*leaf).kv(self.index - 1))
            } else {
                let mut tsp = self.stack.len();
                while tsp > 0 {
                    tsp -= 1;
                    let (nl, ix, _len) = self.stack[tsp];
                    if ix > 0 {
                        return Some((*nl).leaf.kv(ix - 1));
                    }
                }
                None
            }
        }
    }
}

/// Error type for [`CursorMut::insert_before`] and [`CursorMut::insert_after`].
#[derive(Debug, Clone)]
pub struct UnorderedKeyError {}

/// Error returned by [`BTreeMap::try_insert`].
pub struct OccupiedError<'a, K, V, const N: usize, const M: usize>
where
    K: 'a,
    V: 'a,
{
    /// Occupied entry, has the key that was not inserted.
    pub entry: OccupiedEntry<'a, K, V, N, M>,
    /// Value that was not inserted.
    pub value: V,
}

/// Entry in `BTreeMap`, returned by [`BTreeMap::entry`].
pub enum Entry<'a, K, V, const N: usize, const M: usize> {
    /// Vacant entry - map doesn't yet contain key.
    Vacant(VacantEntry<'a, K, V, N, M>),
    /// Occupied entry - map already contains key.
    Occupied(OccupiedEntry<'a, K, V, N, M>),
}
impl<'a, K, V, const N: usize, const M: usize> Entry<'a, K, V, N, M>
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
    pub fn and_modify<F>(mut self, f: F) -> Entry<'a, K, V, N, M>
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
pub struct VacantEntry<'a, K, V, const N: usize, const M: usize> {
    key: K,
    cursor: CursorMut<'a, K, V, N, M>,
}

impl<'a, K, V, const N: usize, const M: usize> VacantEntry<'a, K, V, N, M>
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
pub struct OccupiedEntry<'a, K, V, const N: usize, const M: usize> {
    cursor: CursorMut<'a, K, V, N, M>,
}

impl<'a, K, V, const N: usize, const M: usize> OccupiedEntry<'a, K, V, N, M>
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
        std::mem::replace(self.get_mut(), value)
    }
}

#[test]
fn exp_mem_test() {
    const N: usize = 47;
    const M: usize = N + 1;
    let n = 1000;
    let mut map = BTreeMap::<u64, u64, N, M>::new();
    for i in 0..n {
        map.insert(i * 2, i * 2);
    }
    for i in 0..n {
        map.insert(i * 2 + 1, i * 2 + 1);
    }
    crate::print_memory();
    println!("Required memory: {} bytes", n * 32);
    println!("size of Leaf={}", std::mem::size_of::<Leaf<u64, u64, N>>());
    println!(
        "size of NonLeaf={}",
        std::mem::size_of::<NonLeaf<u64, u64, N, M>>()
    );
    map.print_clen();
}

#[test]
fn std_mem_test() {
    let n = 1000;
    let mut map = std::collections::BTreeMap::<u64, u64>::new();
    for i in 0..n {
        map.insert(i * 2, i * 2);
    }
    for i in 0..n {
        map.insert(i * 2 + 1, i * 2 + 1);
    }
    crate::print_memory();
}

#[test]
fn general_test() {
    let n: usize = 30;
    let mut map = BTreeMap::<usize, usize, 5, 6>::new();
    for i in (0..n).rev() {
        map.insert(i, i);
        assert_eq!(i, *map.get(&i).unwrap());
    }
    assert_eq!(map.len(), n);
    for i in 0..n {
        let v = map.get_mut(&i).unwrap();
        assert_eq!(*v, i);
    }

    assert_eq!(map.pop_last(), Some((n - 1, n - 1)));
    assert_eq!(map.len(), n - 1);
    assert_eq!(map.pop_first(), Some((0, 0)));
    assert_eq!(map.len(), n - 2);

    let mut c = map.lower_bound_mut(Bound::Unbounded);

    for i in 1..n - 1 {
        let (k, v) = c.next().unwrap();
        assert_eq!((*k, *v), (i, i));
    }
    assert_eq!(c.next(), None);

    // let mut c = map.upper_bound_mut(Bound::Unbounded);

    for i in (1..n - 1).rev() {
        let (k, v) = c.prev().unwrap();
        assert_eq!((*k, *v), (i, i));
    }
    for i in 1..(n - 1) {
        let v = map.remove(&i).unwrap();
        assert_eq!(v, i);
    }
    assert_eq!(map.len(), 0);

    let mut c = map.lower_bound_mut(Bound::Unbounded);

    for i in 0..50 {
        c.insert_before(i, i).unwrap();
        let (k, v) = c.peek_prev().unwrap();
        assert_eq!((*k, *v), (i, i));
        assert_eq!(c.peek_next(), None);
    }
    assert_eq!(map.len(), 50);

    for i in 0..50 {
        assert_eq!(map.get(&i).unwrap(), &i);
    }

    for (i, x) in map.iter().enumerate() {
        assert_eq!(&i, x.0);
        assert_eq!(&i, x.1);
    }

    for (i, x) in map.iter_mut().enumerate() {
        assert_eq!(&i, x.0);
        assert_eq!(&i, x.1);
    }

    let mut c = map.lower_bound_mut(Bound::Unbounded);
    for i in 0..50 {
        let (k, v) = c.remove_next().unwrap();
        assert_eq!((k, v), (i, i));
    }

    assert_eq!(map.len(), 0);
}
