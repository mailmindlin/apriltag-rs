use std::{alloc::AllocError, mem::MaybeUninit};

use datasize::DataSize;

/// Entry for [LookupTable]
#[derive(Copy, Clone, DataSize)]
struct Entry<V> {
    key: u64,
    value: V,
}

impl<V> Entry<V> {
	#[inline]
	const fn empty() -> Self where V: ~const Default {
		Self {
			key: u64::MAX,
            value: Default::default(),
		}
	}

	#[inline]
	fn is_empty(&self) -> bool {
		self.key == u64::MAX
	}
}


struct BucketIter {
	current: usize,
	capacity: usize,
}

impl Iterator for BucketIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
		let res = self.current;
        self.current = (self.current + 1) % self.capacity;
		//TODO: detect complete cycles (currently they will just spin forever)
		Some(res)
    }
}

#[derive(DataSize)]
pub(super) struct LookupTable<V> {
    entries: Box<[Entry<V>]>,
}

impl<V> LookupTable<V> {
    /// Create a LookupTable with given table capacity
	pub(super) fn with_capacity(capacity: usize) -> Result<Self, AllocError> where V: Default + Copy {
		let entries = {
			let mut entries = Box::try_new_zeroed_slice(capacity)?;
					// println!("Failed to allocate hamming decode table for family {}: {:?}", family.name, e);
			
			entries.fill(MaybeUninit::new(Entry::empty()));

			unsafe { entries.assume_init() }
		};

		Ok(Self {
			entries,
		})
	}

    fn bucket_iter(&self, initial_value: u64) -> BucketIter {
		let capacity = self.entries.len();
		let current = initial_value as usize % capacity;
		BucketIter {
			current,
			capacity,
		}
	}

    pub(super) fn add(&mut self, key: u64, value: V) -> Result<(), V> {
        // Make sure we don't try to insert the 'empty' marker
        assert_ne!(key, u64::MAX, "Cannot insert key 2**64");
        let bucket = self.bucket_iter(key)
            .find(|bucket| self.entries[*bucket].is_empty());

        match bucket {
            Some(bucket) => {
                #[cfg(debug_assertions)]
                debug_assert!(self.entries[bucket].is_empty(), "Bucket is not empty");

                self.entries[bucket] = Entry {
                    key,
                    value,
                };
                Ok(())
            }
            None => {
                Err(value)
            }
        }
    }

    pub(super) fn get(&self, key: u64) -> Option<&V> {
        for bucket in self.bucket_iter(key) {
            println!("Test bucket {}", bucket);
            let entry = &self.entries[bucket];
            if entry.is_empty() {
                println!("qd End");
                break;
            }

            println!("\trcode {key} vs {}", entry.key);
            if entry.key == key {
                return Some(&entry.value);
            }
        }
        None
    }

    pub(super) fn stats(&self) -> (f64, usize) {
        let mut longest_run = 0;
        let mut run = 0usize;
        let mut run_sum = 0;
        let mut run_count = 0usize;

        // This accounting code doesn't check the last possible run that
        // occurs at the wrap-around. That's pretty insignificant.
        for entry in self.entries.iter() {
            if entry.is_empty() {
                if run > 0 {
                    run_sum += run;
                    run_count += 1;
                }
                run = 0;
            } else {
                run += 1;
                if run > longest_run {
                    longest_run = run;
                }
            }
        }

        let avg_run = (run_sum as f64) / (run_count as f64);

        (avg_run, longest_run)
    }
}