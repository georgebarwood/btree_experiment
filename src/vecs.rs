use std::{
    alloc,
    alloc::Layout,
    cmp::Ordering,
    fmt,
    fmt::Debug,
    mem,
    ops::{Deref, DerefMut},
    ptr,
    ptr::NonNull,
};

/// Basic vec, does not have own capacity or length, just a pointer to memory.
/// Kind-of cribbed from <https://doc.rust-lang.org/nomicon/vec/vec-final.html>.
struct BasicVec<T> {
    p: NonNull<T>,
}

unsafe impl<T: Send> Send for BasicVec<T> {}
unsafe impl<T: Sync> Sync for BasicVec<T> {}

impl<T> Default for BasicVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> BasicVec<T> {
    /// Construct new `BasicVec`.
    pub fn new() -> Self {
        Self {
            p: NonNull::dangling(),
        }
    }

    /// Get mutable raw pointer to specified element.
    /// # Safety
    /// index must be < set capacity.
    #[inline]
    pub unsafe fn ix(&self, index: usize) -> *mut T {
        self.p.as_ptr().add(index)
    }

    /// Set capacity ( allocate or reallocate memory ).
    /// # Safety
    ///
    /// `oa` must be the previous alloc set (0 if no alloc has yet been set).
    pub unsafe fn set_alloc(&mut self, oa: usize, na: usize) {
        if mem::size_of::<T>() == 0 {
            return;
        }
        if na == 0 {
            self.free(oa);
            return;
        }
        let new_layout = Layout::array::<T>(na).unwrap();

        let new_ptr = if oa == 0 {
            alloc::alloc(new_layout)
        } else {
            let old_layout = Layout::array::<T>(oa).unwrap();
            let old_ptr = self.p.as_ptr().cast::<u8>();
            alloc::realloc(old_ptr, old_layout, new_layout.size())
        };

        // If allocation fails, `new_ptr` will be null, in which case we abort.
        self.p = match NonNull::new(new_ptr.cast::<T>()) {
            Some(p) => p,
            None => alloc::handle_alloc_error(new_layout),
        };
    }

    /// Free memory.
    /// # Safety
    ///
    /// The capacity must be the last capacity set.
    pub unsafe fn free(&mut self, oa: usize) {
        let elem_size = mem::size_of::<T>();
        if oa != 0 && elem_size != 0 {
            alloc::dealloc(
                self.p.as_ptr().cast::<u8>(),
                Layout::array::<T>(oa).unwrap(),
            );
        }
        self.p = NonNull::dangling();
    }

    /// Set value.
    /// # Safety
    ///
    /// ix must be < capacity, and the element must be unset.
    #[inline]
    pub unsafe fn set(&mut self, i: usize, elem: T) {
        ptr::write(self.ix(i), elem);
    }

    /// Get value.
    /// # Safety
    ///
    /// ix must be less < capacity, and the element must have been set.
    #[inline]
    pub unsafe fn get(&mut self, i: usize) -> T {
        ptr::read(self.ix(i))
    }

    /// Get whole as slice.
    /// # Safety
    ///
    /// len must be <= capacity and 0..len elements must have been set.
    #[inline]
    pub unsafe fn slice(&self, len: usize) -> &[T] {
        std::slice::from_raw_parts(self.p.as_ptr(), len)
    }

    /// Get whole as mut slice.
    /// # Safety
    ///
    /// len must be <= capacity and 0..len elements must have been set.
    #[inline]
    pub unsafe fn slice_mut(&mut self, len: usize) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.p.as_ptr(), len)
    }

    /// Move elements.
    /// # Safety
    ///
    /// The set status of the elements changes in the obvious way. from, to and len must be in range.
    pub unsafe fn move_self(&mut self, from: usize, to: usize, len: usize) {
        ptr::copy(self.ix(from), self.ix(to), len);
    }

    /// Move elements from another `BasicVec`.
    /// # Safety
    ///
    /// The set status of the elements changes in the obvious way. from, to and len must be in range.
    pub unsafe fn move_from(&mut self, from: usize, src: &mut Self, to: usize, len: usize) {
        ptr::copy_nonoverlapping(src.ix(from), self.ix(to), len);
    }
}

/// In debug mode or feature unsafe-optim not enabled, same as assert! otherwise does nothing.
#[cfg(any(debug_assertions, not(feature = "unsafe-optim")))]
macro_rules! safe_assert {
    ( $cond: expr ) => {
        assert!($cond)
    };
}

/// In debug mode or feature unsafe-optim not enabled, same as assert! otherwise does nothing.
#[cfg(all(not(debug_assertions), feature = "unsafe-optim"))]
macro_rules! safe_assert {
    ( $cond: expr ) => {};
}

/// Vec with limited capacity that allocates incrementally and trims when split.
pub(crate) struct ShortVec<T> {
    len: u16,   // Current length.
    alloc: u16, // Currently allocated.
    cap: u16,   // Maximum capacity ( never allocate more than this ).
    v: BasicVec<T>,
}

impl<T> Default for ShortVec<T> {
    fn default() -> Self {
        Self::new(u16::MAX as usize)
    }
}

impl<T> Drop for ShortVec<T> {
    fn drop(&mut self) {
        let mut len = self.len as usize;
        while len > 0 {
            len -= 1;
            unsafe {
                self.v.get(len);
            }
        }
        unsafe {
            self.v.free(self.alloc as usize);
        }
    }
}

impl<T> ShortVec<T> {
    pub fn new(cap: usize) -> Self {
        safe_assert!(cap <= u16::MAX as usize);
        let v = BasicVec::new();
        Self {
            len: 0,
            alloc: 0,
            cap: cap as u16,
            v,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    fn allocate(&mut self, amount: usize) {
        safe_assert!(amount <= self.cap as usize);
        if amount > self.alloc as usize {
            self.increase_alloc(amount);
        }
    }

    fn increase_alloc(&mut self, amount: usize) {
        let mut na = amount + 5;
        if na + 4 > self.cap as usize {
            na = self.cap as usize;
        }
        unsafe {
            self.v.set_alloc(self.alloc as usize, na);
        }
        self.alloc = na as u16;
    }

    fn trim(&mut self) {
        let na = self.len();
        if self.alloc as usize > na {
            unsafe {
                self.v.set_alloc(self.alloc as usize, na);
            }
            self.alloc = na as u16;
        }
    }
    #[inline]
    pub fn push(&mut self, value: T) {
        self.allocate(self.len() + 1);
        unsafe {
            self.v.set(self.len(), value);
        }
        self.len += 1;
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            unsafe { Some(self.v.get(self.len())) }
        }
    }

    pub fn insert(&mut self, at: usize, value: T) {
        self.allocate(self.len() + 1);
        unsafe {
            if at < self.len() {
                self.v.move_self(at, at + 1, self.len() - at);
            }
            self.v.set(at, value);
            self.len += 1;
        }
    }

    pub fn remove(&mut self, at: usize) -> T {
        safe_assert!(at < self.len());
        unsafe {
            let result = self.v.get(at);
            self.v.move_self(at + 1, at, self.len() - at - 1);
            self.len -= 1;
            self.trim();
            result
        }
    }

    pub fn split_off(&mut self, at: usize) -> Self {
        safe_assert!(at < self.len());
        let len = self.len() - at;
        let mut result = Self::new(self.cap as usize);
        result.allocate(len);
        unsafe {
            result.v.move_from(at, &mut self.v, 0, len);
        }
        result.len = len as u16;
        self.len -= len as u16;
        self.trim();
        result
    }

    /// Get reference to ith element.
    #[inline]
    pub fn ix(&self, i: usize) -> &T {
        safe_assert!(i < self.len());
        unsafe { &*self.v.ix(i) }
    }

    /// Get mutable reference to ith element.
    #[inline]
    pub fn ixm(&mut self, i: usize) -> &mut T {
        safe_assert!(i < self.len());
        unsafe { &mut *self.v.ix(i) }
    }
}

impl<T> Deref for ShortVec<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        let len: usize = ShortVec::len(self);
        unsafe { self.v.slice(len) }
    }
}

impl<T> DerefMut for ShortVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        let len: usize = ShortVec::len(self);
        unsafe { self.v.slice_mut(len) }
    }
}

impl<T> fmt::Debug for ShortVec<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T> IntoIterator for ShortVec<T> {
    type Item = T;
    type IntoIter = ShortVecIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        ShortVecIter { start: 0, v: self }
    }
}

pub(crate) struct ShortVecIter<T> {
    start: usize,
    v: ShortVec<T>,
}

impl<T> Iterator for ShortVecIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.start == self.v.len() {
            None
        } else {
            let ix = self.start;
            self.start += 1;
            Some(unsafe { self.v.v.get(ix) })
        }
    }
}
impl<T> DoubleEndedIterator for ShortVecIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start == self.v.len() {
            None
        } else {
            self.v.len -= 1;
            Some(unsafe { self.v.v.get(self.v.len()) })
        }
    }
}
impl<T> Drop for ShortVecIter<T> {
    fn drop(&mut self) {
        while self.len() > 0 {
            self.next();
        }
        self.v.len = 0;
    }
}
impl<T> ExactSizeIterator for ShortVecIter<T> {
    fn len(&self) -> usize {
        self.v.len() - self.start
    }
}

use std::marker::PhantomData;

/// Vector of (key,value) pairs, keys stored separately from values for cache efficient search.
pub struct PairVec<K, V> {
    p: NonNull<u8>,
    len: u16,      // Current length
    alloc: u16,    // Allocated
    capacity: u16, // Maximum capacity
    _pd: PhantomData<(K, V)>,
}

impl<K, V> Default for PairVec<K, V> {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<K, V> Drop for PairVec<K, V> {
    fn drop(&mut self) {
        while self.len != 0 {
            self.pop();
        }
        self.trim();
    }
}

impl<K, V> PairVec<K, V> {
    /// ...
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity as u16;
        Self {
            p: NonNull::dangling(),
            len: 0,
            alloc: 0,
            capacity,
            _pd: PhantomData,
        }
    }

    /// ...
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// ...
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// ...
    pub fn full(&self) -> bool {
        self.len == self.capacity
    }

    /// ...
    pub fn cap(&self) -> usize {
        self.capacity as usize
    }

    #[inline]
    unsafe fn layout(amount: usize) -> (Layout, usize) {
        let layout = Layout::array::<K>(amount).unwrap();
        let (layout, off) = layout.extend(Layout::array::<V>(amount).unwrap()).unwrap();
        (layout, off)
    }

    ///
    pub fn trim(&mut self) {
        self.alloc(self.len());
    }

    /// ...
    pub fn allocate(&mut self, mut amount: usize) {
        if amount > self.capacity as usize {
            amount = self.capacity as usize;
        }
        self.alloc(amount);
    }

    /// ...
    fn alloc(&mut self, amount: usize) {
        unsafe {
            let (old_layout, old_off) = Self::layout(self.alloc as usize);

            let np = if amount == 0 {
                NonNull::dangling()
            } else {
                let (layout, off) = Self::layout(amount);
                let np = alloc::alloc(layout);
                let np = match NonNull::new(np.cast::<u8>()) {
                    Some(np) => np,
                    None => alloc::handle_alloc_error(layout),
                };

                // Copy keys and values from old allocation to new allocation.
                if self.len > 0 {
                    let from = self.p.as_ptr().cast::<K>();
                    let to = np.as_ptr().cast::<K>();
                    ptr::copy_nonoverlapping(from, to, self.len as usize);

                    let from = self.p.as_ptr().add(old_off).cast::<V>();
                    let to = np.as_ptr().add(off).cast::<V>();
                    ptr::copy_nonoverlapping(from, to, self.len as usize);
                }
                np
            };

            // Free the old allocation.
            if self.alloc > 0 {
                alloc::dealloc(self.p.as_ptr(), old_layout);
            }
            self.alloc = amount as u16;
            self.p = np;
        }
    }

    /// ...
    pub fn split_off(&mut self, at: usize, r: usize) -> Self {
        safe_assert!(at <= self.len());
        safe_assert!(r <= 1);
        let len = self.len() - at;
        let mut result = Self::new(self.capacity as usize);
        result.allocate(len + r * 5);
        unsafe {
            let (kf, vf) = self.ixmp(at);
            let (kt, vt) = result.ixmp(0);
            ptr::copy_nonoverlapping(kf, kt, len);
            ptr::copy_nonoverlapping(vf, vt, len);
        }
        result.len = len as u16;
        self.len -= len as u16;
        self.allocate(self.len() + (1 - r)*5);
        result
    }

    /// ...
    pub fn insert(&mut self, at: usize, (key, value): (K, V)) {
        safe_assert!(self.len < self.capacity);
        safe_assert!(at <= self.len());
        if self.alloc == self.len {
            self.allocate(self.len() + 5);
        }
        unsafe {
            let n = self.len() - at;
            let (kp, vp) = self.ixmp(at);
            if n > 0 {
                ptr::copy(kp, kp.add(1), n);
                ptr::copy(vp, vp.add(1), n);
            }
            kp.write(key);
            vp.write(value);
            self.len += 1
        }
    }

    /// ...
    pub fn remove(&mut self, at: usize) -> (K, V) {
        safe_assert!(at < self.len());
        unsafe {
            let n = self.len() - at - 1;
            let (kp, vp) = self.ixmp(at);
            let result = (kp.read(), vp.read());
            if n > 0 {
                ptr::copy(kp.add(1), kp, n);
                ptr::copy(vp.add(1), vp, n);
            }
            self.len -= 1;
            result
        }
    }

    /// ...
    pub fn push(&mut self, (key, value): (K, V)) {
        safe_assert!(self.len < self.capacity);
        if self.alloc == self.len {
            self.allocate(self.len() + 5);
        }
        unsafe {
            let (kp, vp) = self.ixmp(self.len());
            kp.write(key);
            vp.write(value);
            self.len += 1;
        }
    }
    /// ...
    pub fn pop(&mut self) -> Option<(K, V)> {
        unsafe {
            if self.len == 0 {
                return None;
            }
            self.len -= 1;
            let (kp, vp) = self.ixmp(self.len());
            Some((kp.read(), vp.read()))
        }
    }

    /// Same as `binary_search_by`, but for some obscure reason this seems to be faster.
    #[inline]
    pub fn search<F>(&self, f: F) -> Result<usize, usize>
    where
        F: FnMut(&K) -> Ordering,
    {
        self.search_to(self.len(), f)
    }

    /// ...
    pub fn search_to<F>(&self, n: usize, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&K) -> Ordering,
    {
        safe_assert!(n <= self.len());
        let (mut i, mut j) = (0, n);
        while i < j {
            let m = (i + j) / 2;
            match f(self.ixk(m)) {
                Ordering::Equal => return Ok(m),
                Ordering::Less => i = m + 1,
                Ordering::Greater => j = m,
            }
        }
        Err(i)
    }

    /// ...
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        unsafe {
            let mut i = 0;
            let mut r = 0;
            while i < self.len() {
                let (k, v) = self.ixm(i);
                if f(k, v) {
                    if r != i {
                        let kv = self.get(i);
                        self.set(r, kv);
                    }
                    r += 1;
                } else {
                    self.get(i);
                }
                i += 1;
            }
            self.len -= (i - r) as u16;
            self.trim();
        }
    }

    #[inline]
    unsafe fn get(&mut self, i: usize) -> (K, V) {
        let (kp, vp) = self.ixmp(i);
        (kp.read(), vp.read())
    }

    #[inline]
    unsafe fn set(&mut self, i: usize, (k, v): (K, V)) {
        let (kp, vp) = self.ixmp(i);
        kp.write(k);
        vp.write(v);
    }

    #[inline]
    pub unsafe fn ixmp(&mut self, i: usize) -> (*mut K, *mut V) {
        let (_, off) = Self::layout(self.alloc as usize);
        let kp = self.p.as_ptr().cast::<K>().add(i);
        let vp = self.p.as_ptr().add(off).cast::<V>().add(i);
        (kp, vp)
    }

    #[inline]
    pub unsafe fn ixp(&self, i: usize) -> (*const K, *const V) {
        let (_, off) = Self::layout(self.alloc as usize);
        let kp = self.p.as_ptr().cast::<K>().add(i);
        let vp = self.p.as_ptr().add(off).cast::<V>().add(i);
        (kp, vp)
    }

    #[inline]
    pub fn ixk(&self, i: usize) -> &K {
        safe_assert!(i < self.len());
        unsafe {
            let kp = self.p.as_ptr().cast::<K>().add(i);
            &*kp
        }
    }

    #[inline]
    pub fn ixmv(&mut self, i: usize) -> &mut V {
        safe_assert!(i < self.len());
        unsafe {
            let (_kp, vp) = self.ixmp(i);
            &mut *vp
        }
    }

    #[inline]
    pub fn ix(&self, i: usize) -> (&K, &V) {
        safe_assert!(i < self.len());
        unsafe {
            let (kp, vp) = self.ixp(i);
            (&*kp, &*vp)
        }
    }

    #[inline]
    pub fn ixm(&mut self, i: usize) -> (&K, &mut V) {
        safe_assert!(i < self.len());
        unsafe {
            let (kp, vp) = self.ixmp(i);
            (&*kp, &mut *vp)
        }
    }

    #[inline]
    pub fn ixbm(&mut self, i: usize) -> (&mut K, &mut V) {
        safe_assert!(i < self.len());
        unsafe {
            let (kp, vp) = self.ixmp(i);
            (&mut *kp, &mut *vp)
        }
    }

    pub fn iter(&self) -> IterPairVec<K, V> {
        IterPairVec {
            v: self,
            ix: 0,
            ixb: self.len(),
        }
    }
    /// ...
    pub fn range(&self, x: usize, y: usize) -> IterPairVec<K, V> {
        safe_assert!(x <= y && y <= self.len());
        IterPairVec {
            v: self,
            ix: x,
            ixb: y,
        }
    }
    /// ...
    pub fn iter_mut(&mut self) -> IterMutPairVec<K, V> {
        let ixb = self.len();
        IterMutPairVec {
            v: self,
            ix: 0,
            ixb,
        }
    }
    /// ...
    pub fn range_mut(&mut self, x: usize, y: usize) -> IterMutPairVec<K, V> {
        safe_assert!(x <= y && y <= self.len());
        IterMutPairVec {
            v: self,
            ix: x,
            ixb: y,
        }
    }
    /// ...
    pub fn into_iter(self) -> IntoIterPairVec<K, V> {
        let ixb = self.len();
        IntoIterPairVec {
            v: self,
            ix: 0,
            ixb,
        }
    }
}

impl<K, V> fmt::Debug for PairVec<K, V>
where
    K: Debug,
    V: Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_map()
            .entries(self.iter().map(|(k, v)| (k, v)))
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct IterPairVec<'a, K, V> {
    v: &'a PairVec<K, V>,
    ix: usize,
    ixb: usize,
}
impl<'a, K, V> IterPairVec<'a, K, V> {
    pub fn len(&self) -> usize {
        self.ixb - self.ix
    }
}
impl<'a, K, V> Iterator for IterPairVec<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.ix == self.ixb {
                return None;
            }
            let (kp, vp) = self.v.ixp(self.ix);
            self.ix += 1;
            Some((&*kp, &*vp))
        }
    }
}
impl<'a, K, V> DoubleEndedIterator for IterPairVec<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.ix == self.ixb {
                return None;
            }
            self.ixb -= 1;
            let (kp, vp) = self.v.ixp(self.ixb);
            Some((&*kp, &*vp))
        }
    }
}

#[derive(Debug)]
pub struct IterMutPairVec<'a, K, V> {
    v: &'a mut PairVec<K, V>,
    ix: usize,
    ixb: usize,
}
impl<'a, K, V> IterMutPairVec<'a, K, V> {
    pub fn len(&self) -> usize {
        self.ixb - self.ix
    }
}
impl<'a, K, V> Iterator for IterMutPairVec<'a, K, V> {
    type Item = (&'a K, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.ix == self.ixb {
                return None;
            }
            let (kp, vp) = self.v.ixmp(self.ix);
            self.ix += 1;
            Some((&mut *kp, &mut *vp))
        }
    }
}
impl<'a, K, V> DoubleEndedIterator for IterMutPairVec<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.ix == self.ixb {
                return None;
            }
            self.ixb -= 1;
            let (kp, vp) = self.v.ixmp(self.ixb);
            Some((&mut *kp, &mut *vp))
        }
    }
}

#[derive(Debug)]
pub struct IntoIterPairVec<K, V> {
    v: PairVec<K, V>,
    ix: usize,
    ixb: usize,
}
impl<K, V> IntoIterPairVec<K, V> {
    pub fn len(&self) -> usize {
        self.ixb - self.ix
    }
}
impl<K, V> Iterator for IntoIterPairVec<K, V> {
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.ix == self.ixb {
                return None;
            }
            let (kp, vp) = self.v.ixmp(self.ix);
            self.ix += 1;
            Some((kp.read(), vp.read()))
        }
    }
}
impl<K, V> DoubleEndedIterator for IntoIterPairVec<K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.ix == self.ixb {
                return None;
            }
            self.ixb -= 1;
            let (kp, vp) = self.v.ixmp(self.ixb);
            Some((kp.read(), vp.read()))
        }
    }
}
impl<K, V> Drop for IntoIterPairVec<K, V> {
    fn drop(&mut self) {
        while self.ix != self.ixb {
            self.next();
        }
        self.v.len = 0;
    }
}

#[test]
fn test_kv() {
    let mut v = PairVec::new(4);
    v.push((2, 2));
    v.insert(0, (1, 1));
    println!("ixm(0) = {:?}", v.ixm(0));
    println!("ixm(1) = {:?}", v.ixm(1));

    for kv in v.iter_mut() {
        println!("kv={:?}", kv);
    }

    //println!("pop = {:?}", v.pop().unwrap());
    //println!("pop = {:?}", v.pop().unwrap());
    println!("remove(0)= {:?}", v.remove(0));

    for kv in v.into_iter() {
        println!("kv={:?}", kv);
    }
}
