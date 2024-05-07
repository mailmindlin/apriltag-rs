use std::{sync::{RwLock, Arc}, mem::MaybeUninit};


type ListInner<T> = Box<[T]>;

struct ListTable<T> {
    boxes: Box<[Arc<ListInner<T>>]>,
    current: RwLock<(ListInner<MaybeUninit<T>>, usize)>,
}

fn box_for_idx(idx: usize) -> (usize, usize) {
    // First box is size 16
    idx.checked_ilog2()
}

impl<T> ListTable<T> {
    fn push(&mut self, value: T) {
        let mut guard = self.current.write().unwrap();
        
    }
}

#[derive(Clone)]
pub(super) struct ListSnapshot<T> {
    boxes: Box<[Arc<ListInner<T>>]>,
    len: usize,
}

impl<T> ListSnapshot<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn iter<'a>(&'a self) -> ListSnapshotIter<'a, T> {
        ListSnapshotIter { snapshot: self, index: 0 }
    }
}

pub(super) struct ListSnapshotIter<'a, T> {
    snapshot: &'a ListSnapshot<T>,
    index: usize,
}

impl<'a, T> Iterator for ListSnapshotIter<'a, T> {
    type Item = &'a T;

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.snapshot.len - self.index;
        (rem, Some(rem))
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.index += n;
        self.next()
    }

    fn next(&mut self) -> Option<Self::Item> {
        let x = self.snapshot.boxes.iter();
        todo!()
    }
}


pub(super) struct WriteList<T> {
    data: RwLock<Arc<ListTable<T>>>,
}

impl<T> WriteList<T> {
    pub(super) fn new() -> Self {

    }
    pub(super) fn push(&self, value: T) {
        let mut data = self.data.read().unwrap();
        let mut inner = data.write().unwrap();
    }
    pub(super) fn len(&self) -> usize {
        let data = self.data.read().unwrap();
        data.len()
    }
    pub(super) fn extend<const N: usize>(&self, values: [T; N]) {
        
    }
    pub(super) fn snapshot(&self) -> ListSnapshot<T> {

    }
}