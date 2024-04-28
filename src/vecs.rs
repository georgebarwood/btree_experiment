use std::{
    alloc,
    alloc::Layout,
    cmp::Ordering,
    fmt, mem,
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
    }

    /// Set value.
    /// # Safety
    ///
    /// ix must be < capacity, and the element must be unset.
    #[inline]
    pub unsafe fn set(&mut self, ix: usize, elem: T) {
        ptr::write(self.ix(ix), elem);
    }

    /// Get value.
    /// # Safety
    ///
    /// ix must be less < capacity, and the element must have been set.
    #[inline]
    pub unsafe fn get(&mut self, ix: usize) -> T {
        ptr::read(self.ix(ix))
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
    pub fn is_empty(&self) -> bool {
        self.len == 0
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

    pub fn split_off(&mut self, at: usize, cap: usize) -> Self {
        safe_assert!(at < self.len());
        let len = self.len() - at;
        let mut result = Self::new(cap);
        result.allocate(len);
        unsafe {
            result.v.move_from(at, &mut self.v, 0, len);
        }
        result.len = len as u16;
        self.len -= len as u16;
        self.trim();
        result
    }

    pub fn retain_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        unsafe {
            let mut i = 0;
            let mut r = 0;
            while i < self.len() {
                if f(&mut *self.v.ix(i)) {
                    if r != i {
                        let v = self.v.get(i);
                        self.v.set(r, v);
                    }
                    r += 1;
                } else {
                    self.v.get(i);
                }
                i += 1;
            }
            self.len -= (i - r) as u16;
            self.trim();
        }
    }

    /// Get reference to ith element.
    #[inline]
    pub fn ix(&self, ix: usize) -> &T {
        safe_assert!(ix < self.len());
        unsafe { &*self.v.ix(ix) }
    }

    /// Get mutable reference to ith element.
    #[inline]
    pub fn ixm(&mut self, ix: usize) -> &mut T {
        safe_assert!(ix < self.len());
        unsafe { &mut *self.v.ix(ix) }
    }

    /// Same as `binary_search_by`, but for some obscure reason this seems to be faster.
    pub fn search<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&T) -> Ordering,
    {
        let (mut i, mut j) = (0, self.len());
        while i < j {
            let m = (i + j) / 2;
            match f(self.ix(m)) {
                Ordering::Equal => return Ok(m),
                Ordering::Less => i = m + 1,
                Ordering::Greater => j = m,
            }
        }
        Err(i)
    }

    pub fn sv_iter(self) -> ShortVecIter<T> {
        ShortVecIter { start: 0, v: self }
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
