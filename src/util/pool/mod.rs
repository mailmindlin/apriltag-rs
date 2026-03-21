use std::{ops::{DerefMut, Deref}, borrow::{Borrow, BorrowMut}, sync::{RwLock, Mutex}};


struct Entry<K, V> {
	key: K,
	value: V,
}


pub(crate) struct KeyedPool<K: PartialEq + Send, V: Send> {
	item: Mutex<Option<Entry<K, V>>>,
}

impl<K: PartialEq + Send, V: Send> Default for KeyedPool<K, V> {
    fn default() -> Self {
        Self { item: Default::default() }
    }
}

pub(crate) struct PoolGuard<'a, K: PartialEq + Send, V: Send> {
	pool: &'a KeyedPool<K, V>,
	entry: Option<Entry<K, V>>,
}

impl<'a, K: PartialEq + Send, V: Send> Drop for PoolGuard<'a, K, V> {
	fn drop(&mut self) {
		if let Some(entry) = self.entry.take() {
			self.pool.insert(entry);
		}
	}
}

impl<'a, K: PartialEq + Send, V: Send> Deref for PoolGuard<'a, K, V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        &self.entry.as_ref().unwrap().value
    }
}

impl<'a, K: PartialEq + Send, V: Send> DerefMut for PoolGuard<'a, K, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.entry.as_mut().unwrap().value
    }
}

impl<'a, K: PartialEq + Send, V: Send> Borrow<V> for PoolGuard<'a, K, V> {
    fn borrow(&self) -> &V {
        self.deref()
    }
}

impl<'a, K: PartialEq + Send, V: Send> BorrowMut<V> for PoolGuard<'a, K, V> {
    fn borrow_mut(&mut self) -> &mut V {
        self.deref_mut()
    }
}

impl<K: PartialEq + Send, V: Send> KeyedPool<K, V> {
	pub fn try_borrow<'a>(&'a self, key: K) -> Option<PoolGuard<'a, K, V>> {
		let value = match self.item.try_lock() {
			Ok(mut wg) => wg.take()?,
			Err(_) => return None,
		};
		Some(PoolGuard {
			pool: self,
			entry: Some(value),
		})
	}

	pub fn offer<'a>(&'a self, key: K, value: V) -> PoolGuard<'a, K, V> {
		PoolGuard { pool: self, entry: Some(Entry { key, value }) }
	}

	fn insert(&self, entry: Entry<K, V>) {
		match self.item.try_lock() {
			Ok(mut wg) => {
				if wg.is_none() {
					*wg = Some(entry);
				}
			},
			Err(_) => {},
		}
	}
}