use concurrent_queue::ConcurrentQueue;
pub(crate) struct Pool<T: Send> {
    max_capacity: usize,
    items: Vec<T>,
    queue: ConcurrentQueue<T>,
}

impl<T: Send> Pool<T> {
    pub fn new() -> Self {
        Self {
            max_capacity: usize::MAX,
            items: Vec::new(),
            queue: ConcurrentQueue::unbounded(),
        }
    }

    pub fn with_capacity(max_capacity: usize) -> Self {
        todo!()
    }
    pub fn checkout<'a>(&'a self) -> PoolRef<'a, T> {
        todo!()
    }
}

pub(crate) struct PoolRef<'a, T> {
    value: &'a mut T,
}

impl<T> Drop for PoolRef<T> {
    fn drop(&mut self) {
        todo!()
    }
}