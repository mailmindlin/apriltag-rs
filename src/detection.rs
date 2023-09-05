use std::{sync::Arc, cmp::Ordering};

use crate::{AprilTagFamily, util::{geom::{Poly2D, Point2D, quad::Quadrilateral}, math::mat::Mat33}, TimeProfile};

/// Represents the detection of a tag.
#[derive(Debug, Clone)]
pub struct AprilTagDetection {
	/// The detected tag's family
	pub family: Arc<AprilTagFamily>,

	/// The decoded ID of the tag
	pub id: usize,

	/// How many error bits were corrected?
	/// 
	/// Note: accepting large numbers of corrected errors leads to greatly
	/// increased false positive rates.
	/// 
	/// Note: As of this implementation, the detector cannot detect tags with
	/// a hamming distance greater than 2.
	pub hamming: u16,

	/// A measure of the quality of the binary decoding process: the
	/// average difference between the intensity of a data bit versus
	/// the decision threshold. Higher numbers roughly indicate better
	/// decodes. This is a reasonable measure of detection accuracy
	/// only for very small tags-- not effective for larger tags (where
	/// we could have sampled anywhere within a bit cell and still
	/// gotten a good detection).
	pub decision_margin: f32,

	/// The 3x3 homography matrix describing the projection from an
	/// "ideal" tag (with corners at (-1,-1), (1,-1), (1,1), and (-1,
	/// 1)) to pixels in the image. This matrix will be freed by
	/// apriltag_detection_destroy.
	pub H: Mat33,

	/// The center of the detection in image pixel coordinates.
	pub center: Point2D,

	/// The corners of the tag in image pixel coordinates. These always
	/// wrap counter-clock wise around the tag.
	pub corners: Quadrilateral,
}

impl AprilTagDetection {
	pub(crate) fn is_same_tag(&self, other: &AprilTagDetection) -> bool {
		self.id == other.id && Arc::ptr_eq(&self.family, &other.family)
	}

	pub(crate) fn cmp_tag(&self, other: &AprilTagDetection) -> Ordering {
		let cmp = Ord::cmp(&self.id, &other.id);
		if cmp != Ordering::Equal {
			return cmp;
		}
		let p1 = Arc::as_ptr(&self.family);
		let p2 = Arc::as_ptr(&other.family);
		p1.cmp(&p2)
	}

	pub(crate) fn cmp_best(&self, other: &AprilTagDetection) -> Ordering {
		// want small hamming
		let cmp = Ord::cmp(&self.hamming, &other.hamming);
		if cmp != Ordering::Equal {
			return cmp;
		}

		// want bigger margins
		let cmp = f32::total_cmp(&self.decision_margin, &other.decision_margin);
		if cmp != Ordering::Equal {
			return cmp.reverse();
		}

		// if we STILL don't prefer one detection over the other, then pick
		// any deterministic criterion.
		for i in 0..4 {
			let cmp = f64::total_cmp(&self.corners[i].x(), &other.corners[i].x());
			if cmp != Ordering::Equal {
				return cmp;
			}
			let cmp = f64::total_cmp(&self.corners[i].y(), &other.corners[i].y());
			if cmp != Ordering::Equal {
				return cmp;
			}
		}

		// at this point, we should only be undecided if the tag detections
		// are *exactly* the same. How would that happen?
		println!("uh oh, no preference for overlappingdetection");

		Ordering::Equal
	}
}

#[derive(Default)]
pub struct Detections {
	pub detections: Vec<AprilTagDetection>,
	pub nquads: u32,
	pub tp: TimeProfile,
}

fn remove_indices<T: Clone>(mut elements: Vec<T>, mut drop_idxs: Vec<usize>) -> Vec<T> {
	// Pop as many from the end as possible (this is cheap)
	drop_idxs.sort_unstable();
	while let Some(idx) = drop_idxs.last() {
		if elements.is_empty() || *idx != elements.len() - 1 {
			break;
		}
		let _idx_pop = drop_idxs.pop();
		#[cfg(debug_assertions)]
		debug_assert_eq!(_idx_pop, Some(elements.len() - 1));
		let _elem_pop = elements.pop();
		#[cfg(debug_assertions)]
		debug_assert!(_elem_pop.is_some());
	}

	assert!(elements.len() >= drop_idxs.len());

	match drop_idxs.len() {
		0 => elements,
		1 => {
			elements.remove(drop_idxs[0]);
			elements
		},
		_ => {
			let len = elements.len();
			
			use core::alloc::Allocator;
			use core::ptr;
			/* INVARIANT: vec.len() > read >= write > write-1 >= 0 */
			struct FillGapOnDrop<'a, T, A: Allocator> {
				/* Offset of the element we want to check if it is duplicate */
				read: usize,
	
				/* Offset of the place where we want to place the non-duplicate
				 * when we find it. */
				write: usize,
	
				/* The Vec that would need correction if `same_bucket` panicked */
				vec: &'a mut Vec<T, A>,
			}
	
			impl<'a, T, A: Allocator> Drop for FillGapOnDrop<'a, T, A> {
				fn drop(&mut self) {
					/* This code gets executed when `same_bucket` panics */
	
					/* SAFETY: invariant guarantees that `read - write`
					 * and `len - read` never overflow and that the copy is always
					 * in-bounds. */
					unsafe {
						let ptr = self.vec.as_mut_ptr();
						let len = self.vec.len();
	
						/* How many items were left when `same_bucket` panicked.
						 * Basically vec[read..].len() */
						let items_left = len.wrapping_sub(self.read);
	
						/* Pointer to first item in vec[write..write+items_left] slice */
						let dropped_ptr = ptr.add(self.write);
						/* Pointer to first item in vec[read..] slice */
						let valid_ptr = ptr.add(self.read);
	
						/* Copy `vec[read..]` to `vec[write..write+items_left]`.
						 * The slices can overlap, so `copy_nonoverlapping` cannot be used */
						ptr::copy(valid_ptr, dropped_ptr, items_left);
	
						/* How many items have been already dropped
						 * Basically vec[read..write].len() */
						let dropped = self.read.wrapping_sub(self.write);
	
						self.vec.set_len(len - dropped);
					}
				}
			}
	
			let mut gap = FillGapOnDrop { read: 0, write: 0, vec: &mut elements };
			let ptr = gap.vec.as_mut_ptr();
	
			/* Drop items while going through Vec, it should be more efficient than
			 * doing slice partition_dedup + truncate */
	
			/* SAFETY: Because of the invariant, read_ptr, prev_ptr and write_ptr
			 * are always in-bounds and read_ptr never aliases prev_ptr */
			for idx in drop_idxs.into_iter() {
                assert!(idx < len);
                debug_assert!(idx >= gap.read);
                let good_count = idx - gap.read;
                let read_ptr = unsafe { ptr.add(gap.read) };
				if good_count > 0 {
					// Compact elements at prev..idx
                    unsafe {
                        let dst = ptr.add(gap.write);
                        ptr::copy(read_ptr, dst, good_count)
                    }
                    gap.read += good_count;
                    gap.write += good_count;
				}
				// Drop element at index
                gap.read += 1;
				unsafe {
					ptr::drop_in_place(ptr.add(idx));
				}
			}
            drop(gap);
            elements
		}
	}
}

#[cfg(test)]
mod test {
    use super::remove_indices;

    #[test]
    fn drop_idxs_noop() {
        let elems = vec![0, 1, 2, 3, 4, 5];
        let idxs = vec![];

        let res = remove_indices(elems.clone(), idxs);
        assert_eq!(res, elems);
    }

    #[test]
    fn drop_idxs_one() {
        let elems = vec![0, 1, 2, 3, 4, 5];
        assert_eq!(vec![1, 2, 3, 4, 5], remove_indices(elems.clone(), vec![0]));
        assert_eq!(vec![0, 1, 2, 3, 5], remove_indices(elems.clone(), vec![4]));
        assert_eq!(vec![0, 1, 3, 4, 5], remove_indices(elems.clone(), vec![2]));
    }

    #[test]
    fn drop_idxs_many() {
        let elems = vec![0, 1, 2, 3, 4, 5];
        assert_eq!(vec![1, 3, 5], remove_indices(elems.clone(), vec![0, 2, 4]));
        assert_eq!(vec![0, 2, 4], remove_indices(elems.clone(), vec![1, 3, 5]));
        assert_eq!(vec![0, 5], remove_indices(elems.clone(), vec![1, 2, 3, 4]));
        assert_eq!(vec![5], remove_indices(elems.clone(), vec![0, 1, 2, 3, 4]));
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn drop_error() {
        remove_indices(vec![1, 2], vec![0, 0]);
    }
}

/// Step 3. Reconcile detections--- don't report the same tag more
/// than once. (Allow non-overlapping duplicate detections.)
pub(super) fn reconcile_detections(mut detections: Vec<AprilTagDetection>) -> Vec<AprilTagDetection> {
	if false {
		return detections;
	}

	// Sort detections by tag, such that the same tags are adjacent
	// This reduces average time complexity from O(n^2) to O(n log n)
	detections.sort_unstable_by(AprilTagDetection::cmp_tag);

	//TODO: should we sort drop_idxs?
	let mut drop_idxs = Vec::new();
	'outer: for (i0, det0) in detections.iter().enumerate() {
		if drop_idxs.contains(&i0) {
			continue;
		}
		let poly0 = Poly2D::from(det0.corners);

		for (i1, det1) in detections.iter().enumerate().skip(i0+1) {
			if !det0.is_same_tag(det1) {
				// They can't be the same detection (we skip outer because detections is already sorted by tag)
				continue 'outer;
			}

			if drop_idxs.contains(&i1) {
				continue;
			}

			let poly1 = Poly2D::from(det1.corners);

			if poly0.overlaps_polygon(&poly1) {
				// the tags overlap. Delete one, keep the other.
				let pref = det0.cmp_best(det1);

				if pref.is_le() {
					// keep det0, destroy det1
					drop_idxs.push(i1);
					continue;
				} else {
					// keep det1, destroy det0
					drop_idxs.push(i0);
					continue 'outer;
				}
			}
		}
	}

    remove_indices(detections, drop_idxs)
	// if !drop_idxs.is_empty() {
	// 	detections
	// 		.into_iter()
	// 		.enumerate()
	// 		.filter(|(idx, _)| !drop_idxs.contains(idx))
	// 		.map(|(_idx, det)| det)
	// 		.collect::<Vec<_>>()
	// } else {
	// 	detections
	// }
}