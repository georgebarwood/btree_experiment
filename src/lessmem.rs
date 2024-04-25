use std::borrow::Borrow;
use std::cmp::Ordering;
use std::mem;
use std::mem::MaybeUninit;
use std::ptr;

/// `BTreeMap` similar to [`std::collections::BTreeMap`] where the node capacity (B) can be specified.
/// N should be an odd number, at least 11, a good value may be 31.
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

    /// Get number of key-value pairs in the map.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Is the map empty?
    pub fn is_empty(&self) -> bool {
        self.len == 0
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
            self.tree.new_root(self.clen, split);
            self.clen = 1;
        }
        self.len += usize::from(x.value.is_none());
        x.value
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
}

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
            Ok(i) => Some((self.ixk(i), self.ixv(i))),
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
            let (med, mut right, mut rlen) = self.split(len);
            if i > N / 2 {
                i -= N / 2 + 1;
                right.insert(rlen, i, key, value);
                rlen += 1;
            } else {
                self.insert(*len as usize, i, key, value);
                *len += 1;
            }
            let right = Tree::L(Box::new(right));
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
            let kp = self.keys.as_ptr().cast::<K>().add(at);
            let rkp = to.keys.as_mut_ptr().cast::<K>();
            ptr::copy_nonoverlapping(kp, rkp, len);
            let vp = self.vals.as_ptr().cast::<V>().add(at);
            let rvp = to.vals.as_mut_ptr().cast::<V>();
            ptr::copy_nonoverlapping(vp, rvp, len);
        }
    }

    fn split(&mut self, len: &mut u8) -> ((K, V), Self, usize) {
        let ix = N / 2 + 1;
        let rlen = *len as usize - ix;
        let mut right = Self::new();
        self.mov(ix, rlen, &mut right);
        *len = ix as u8;
        let med = self.pop_last(len).unwrap();
        (med, right, rlen)
    }

    fn insert(&mut self, len: usize, at: usize, key: K, val: V) {
        assert!(at <= len && len < N);
        let n = len - at;
        unsafe {
            let kp = self.keys.as_mut_ptr().cast::<K>().add(at);
            let vp = self.vals.as_mut_ptr().cast::<V>().add(at);
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
            let kp = self.keys.as_mut_ptr().cast::<K>().add(at);
            let vp = self.vals.as_mut_ptr().cast::<V>().add(at);
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
            let kp = self.keys.as_mut_ptr().cast::<K>().add(at);
            let vp = self.vals.as_mut_ptr().cast::<V>().add(at);
            let k = mem::replace(&mut *kp, kv.0);
            let v = mem::replace(&mut *vp, kv.1);
            (k, v)
        }
    }

    fn push(&mut self, len: usize, key: K, val: V) {
        assert!(len < N);
        unsafe {
            let kp = self.keys.as_mut_ptr().cast::<K>().add(len);
            let vp = self.vals.as_mut_ptr().cast::<V>().add(len);
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
                let kp = self.keys.as_ptr().cast::<K>().add(len);
                let vp = self.vals.as_ptr().cast::<V>().add(len);
                Some((kp.read(), vp.read()))
            }
        }
    }

    fn iter(&self, len: usize) -> LeafIter<K, V, N> {
        LeafIter {
            leaf: self,
            fwd: 0,
            bck: len,
        }
    }

    fn iter_mut(&mut self, len: usize) -> LeafIterMut<K, V, N> {
        let bck = len;
        LeafIterMut {
            leaf: self,
            fwd: 0,
            bck,
        }
    }

    /// Get reference to ith key.
    #[inline]
    fn ixk(&self, ix: usize) -> &K {
        unsafe { &*self.keys.as_ptr().cast::<K>().add(ix) }
    }

    /// Get mutable reference to ith key.
    #[inline]
    fn ixkm(&mut self, ix: usize) -> &mut K {
        unsafe { &mut *self.keys.as_mut_ptr().cast::<K>().add(ix) }
    }

    /// Get reference to ith value.
    #[inline]
    fn ixv(&self, ix: usize) -> &V {
        unsafe { &*self.vals.as_ptr().cast::<V>().add(ix) }
    }

    /// Get mutable reference to ith value.
    #[inline]
    fn ixvm(&mut self, ix: usize) -> &mut V {
        unsafe { &mut *self.vals.as_mut_ptr().cast::<V>().add(ix) }
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
} // end impl Leaf

struct LeafIter<'a, K, V, const N: usize> {
    leaf: &'a Leaf<K, V, N>,
    fwd: usize,
    bck: usize,
}

impl<'a, K, V, const N: usize> Iterator for LeafIter<'a, K, V, N> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        if self.fwd == self.bck {
            None
        } else {
            unsafe {
                let kp = self.leaf.keys.as_ptr().cast::<K>().add(self.fwd);
                let vp = self.leaf.vals.as_ptr().cast::<V>().add(self.fwd);
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
                let kp = self.leaf.keys.as_ptr().cast::<K>().add(self.bck);
                let vp = self.leaf.vals.as_ptr().cast::<V>().add(self.bck);
                Some((&*kp, &*vp))
            }
        }
    }
}

struct LeafIterMut<'a, K, V, const N: usize> {
    leaf: &'a mut Leaf<K, V, N>,
    fwd: usize,
    bck: usize,
}

impl<'a, K, V, const N: usize> Iterator for LeafIterMut<'a, K, V, N> {
    type Item = (&'a mut K, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        if self.fwd == self.bck {
            None
        } else {
            unsafe {
                let kp = self.leaf.keys.as_mut_ptr().cast::<K>().add(self.fwd);
                let vp = self.leaf.vals.as_mut_ptr().cast::<V>().add(self.fwd);
                self.fwd += 1;
                Some((&mut *kp, &mut *vp))
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
                let kp = self.leaf.keys.as_mut_ptr().cast::<K>().add(self.bck);
                let vp = self.leaf.vals.as_mut_ptr().cast::<V>().add(self.bck);
                Some((&mut *kp, &mut *vp))
            }
        }
    }
}

struct LeafIntoIter<K, V, const N: usize> {
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
                let kp = self.leaf.keys.as_ptr().cast::<K>().add(self.fwd);
                let vp = self.leaf.vals.as_ptr().cast::<V>().add(self.fwd);
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
                let kp = self.leaf.keys.as_ptr().cast::<K>().add(self.bck);
                let vp = self.leaf.vals.as_ptr().cast::<V>().add(self.bck);
                Some((kp.read(), vp.read()))
            }
        }
    }
}

unsafe fn ix<T>(p: *const [T], ix: usize) -> *const T {
    p.cast::<T>().add(ix)
}

unsafe fn ixm<T>(p: *mut [T], ix: usize) -> *mut T {
    p.cast::<T>().add(ix)
}

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
            Ok(i) => Some((self.leaf.ixk(i), self.leaf.ixv(i))),
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
            Ok(i) => Some(self.remove_at(len, i)),
            Err(i) => self.child_remove(i, key),
        }
    }

    fn child_is_leaf(&self) -> bool {
        match self.c {
            CA::L(_) => true,
            CA::NL(_) => false,
        }
    }

    fn split(&mut self, len: &mut u8) -> Split<K, V, N, M> {
        assert!(*len as usize == N);
        let mut right = Self::new(self.child_is_leaf());
        let at = N / 2 + 1;
        let rlen = N - at;
        self.leaf.mov(at, rlen, &mut right.leaf);
        *len = at as u8;
        let med = self.leaf.pop_last(len).unwrap();

        let at = M / 2;
        let n = M / 2;
        unsafe {
            match &mut self.c {
                CA::L(leaf) => {
                    let from = ix(leaf.as_ptr(), at);
                    match &mut right.c {
                        CA::L(leaf) => {
                            let to = ixm(leaf.as_mut_ptr(), 0);
                            ptr::copy_nonoverlapping(from, to, n);
                        }
                        CA::NL(_) => panic!(),
                    }
                }
                CA::NL(nl) => {
                    let from = ix(nl.as_ptr(), at);
                    match &mut right.c {
                        CA::NL(nl) => {
                            let to = ixm(nl.as_mut_ptr(), 0);
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
        (med, Tree::NL(Box::new(right)), rlen)
    }

    fn child_free(&mut self, at: usize) {
        let clen = &mut self.clen[at];
        unsafe {
            match &mut self.c {
                CA::L(leaf) => {
                    let p = ixm(leaf.as_mut_ptr(), at);
                    (*p).free(clen);
                    let _ = p.read();
                }
                CA::NL(nl) => {
                    let p = ixm(nl.as_mut_ptr(), at);
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
                CA::L(leaf) => (*ixm(leaf.as_mut_ptr(), at)).pop_first(clen),
                CA::NL(nl) => (*ixm(nl.as_mut_ptr(), at)).pop_first(clen),
            }
        }
    }

    fn child_pop_last(&mut self, at: usize) -> Option<(K, V)> {
        let clen = &mut self.clen[at];
        unsafe {
            match &mut self.c {
                CA::L(leaf) => (*ixm(leaf.as_mut_ptr(), at)).pop_last(clen),
                CA::NL(nl) => (*ixm(nl.as_mut_ptr(), at)).pop_last(clen),
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
                CA::L(leaf) => (*ix(leaf.as_ptr(), at)).get(clen, key),
                CA::NL(nl) => (*ix(nl.as_ptr(), at)).get(clen, key),
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
                CA::L(leaf) => (*ix(leaf.as_ptr(), at)).get_key_value(clen, key),
                CA::NL(nl) => (*ix(nl.as_ptr(), at)).get_key_value(clen, key),
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
                CA::L(leaf) => (*ixm(leaf.as_mut_ptr(), at)).get_mut(clen, key),
                CA::NL(nl) => (*ixm(nl.as_mut_ptr(), at)).get_mut(clen, key),
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
                CA::L(leaf) => (*ixm(leaf.as_mut_ptr(), at)).insert_kv(clen, key, x),
                CA::NL(nl) => (*ixm(nl.as_mut_ptr(), at)).insert_kv(clen, key, x),
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
                CA::L(leaf) => (*ixm(leaf.as_mut_ptr(), at)).remove_entry(clen, key),
                CA::NL(nl) => (*ixm(nl.as_mut_ptr(), at)).remove_entry(clen, key),
            }
        }
    }

    fn remove_at(&mut self, len: &mut u8, i: usize) -> (K, V) {
        if let Some(x) = self.child_pop_last(i) {
            self.leaf.replace(i, x)
        } else {
            self.remove_child(len, i);
            self.leaf.remove(len, i)
        }
    }

    fn push_child(&mut self, len: usize, child: Tree<K, V, N, M>) {
        assert!(len < M);
        unsafe {
            match &mut self.c {
                CA::L(leaf) => {
                    let p = ixm(leaf.as_mut_ptr(), len);
                    p.write(child.leaf());
                }
                CA::NL(nl) => {
                    let p = ixm(nl.as_mut_ptr(), len);
                    p.write(child.non_leaf());
                }
            }
        }
    }

    fn insert_child(&mut self, len: usize, at: usize, clen: usize, child: Tree<K, V, N, M>) {
        assert!(at <= len && len < M);
        unsafe {            
            let n = len - at;
            match &mut self.c {
                CA::L(leaf) => {
                    let p = ixm(leaf.as_mut_ptr(), at);
                    if n > 0 {
                        ptr::copy(p, p.add(1), n);
                    }
                    p.write(child.leaf());
                }
                CA::NL(nl) => {
                    let p = ixm(nl.as_mut_ptr(), at);
                    if n > 0 {
                        ptr::copy(p, p.add(1), n);
                    }
                    p.write(child.non_leaf());
                }
            }
            let mut i = len;
            while i > at + 1 {
                i -= 1;
                self.clen[i] = self.clen[i - 1];
            }
            self.clen[at] = clen as u8;
        }
    }

    fn remove_child(&mut self, len: &mut u8, at: usize) {
        assert!(at < *len as usize);
        self.child_free(at);
        unsafe {
            *len -= 1;
            let n = *len as usize - at - 1;
            match &mut self.c {
                CA::L(leaf) => {
                    let p = ixm(leaf.as_mut_ptr(), at);
                    if n > 0 {
                        ptr::copy(p, p.add(1), n);
                    }
                }
                CA::NL(nl) => {
                    let p = ixm(nl.as_mut_ptr(), at);
                    if n > 0 {
                        ptr::copy(p, p.add(1), n);
                    }
                }
            }
            for i in at..at + n {
                self.clen[i] = self.clen[i + 1];
            }
        }
    }
} // end impl NonLeaf

#[test]
fn test() {
    let n = 200;
    let mut map = BTreeMap::<i32, i32, 5, 6>::new();
    for i in 0..n {
        println!("inserting {}", i);
        map.insert(i, i * i);
        assert_eq!(i * i, *map.get(&i).unwrap());
    }
    println!("map len={}", map.len());
    for i in 0..n {
        let v = map.get_mut(&i).unwrap();
        *v += 1;
        println!("v[{}]={:?}", i, v);
    }
    println!("pop_last={:?}", map.pop_last());
    println!("map len={}", map.len());
    println!("pop_first={:?}", map.pop_first());
    println!("map len={}", map.len());
}