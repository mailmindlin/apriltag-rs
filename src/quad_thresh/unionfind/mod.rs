use crate::{util::ImageY8, detector::DetectorConfig};
use rayon::prelude::*;

pub(crate) use self::uf2d::UnionFind2D;

mod uf2d;
mod atomic;
mod reference;

pub(super) type UnionFindId = u32;

pub(crate) trait UnionFind<I> {
    type Id;

    /// Get set representative and cardinality
    fn get_set(&mut self, index: I) -> (Self::Id, u32);

    fn index_to_id(&self, idx: I) -> Self::Id;

    fn connect(&mut self, a: I, b: I) -> bool {
        let id_a = self.index_to_id(a);
        let id_b = self.index_to_id(b);
        self.connect_ids(id_a, id_b)
    }

    fn connect_ids(&mut self, a: Self::Id, b: Self::Id) -> bool;
}

pub(crate) trait UnionFindStatic<I>: UnionFind<I> {
    fn get_set_static(&self, index: I) -> (Self::Id, u32);
    fn get_set_hops(&self, index: I) -> usize;
}

pub(super) trait UnionFindAtomic<I>: UnionFind<I> + UnionFindStatic<I> {
    fn connect_atomic(&self, a: I, b: I) -> bool {
        let id_a = self.index_to_id(a);
        let id_b = self.index_to_id(b);
        self.connect_ids_atomic(id_a, id_b)
    }

    fn connect_ids_atomic(&self, a: Self::Id, b: Self::Id) -> bool;
}

fn do_unionfind_line2(uf: &impl UnionFindAtomic<(u32, u32), Id = u32>, im: &ImageY8, y: usize) {
    assert!(y > 0);
	// debug_assert_eq!(im.width(), uf.width as usize);
	// #[cfg(debug_assertions)]
	// debug_assert_eq!(im.height(), uf.height as usize);

    let mut v_0_m1 = im[(0, y-1)];
    let mut v_1_m1 = im[(1, y-1)];
    let mut v = im[(0,y)];

	let w = im.width();

	let row = im.row(y);
	let row_up = im.row(y-1);

	for x in 1..(w - 1) {
        let v_m1_m1 = v_0_m1;
        v_0_m1 = v_1_m1;
        v_1_m1 = im[(x + 1, y - 1)];
        let v_m1_0 = v;
        v = row[x];

        if v == 127 {
			continue;
		}

		let idx_xy = uf.index_to_id((x as _, y as _));
		let idx_up = uf.index_to_id((x as _, y as u32 - 1));

        // (dx,dy) pairs for 8 connectivity:
        // (-1, -1)    (0, -1)    (1, -1)
        // (-1, 0)    (REFERENCE)
		if row[x-1] == v {
			uf.connect_ids_atomic(idx_xy, idx_xy - 1);
		}

        if x == 1 || !((v_m1_0 == v_m1_m1) && (v_m1_m1 == v_0_m1)) {
			if row_up[x] == v {
				uf.connect_ids_atomic(idx_xy, idx_up);
			}
        }

        if v == 255 {
            if x == 1 || !(v_m1_0 == v_m1_m1 || v_0_m1 == v_m1_m1) {
				if row_up[x-1] == v {
					uf.connect_ids_atomic(idx_xy, idx_up - 1);
				}
            }
            if v_0_m1 != v_1_m1 {
				if row_up[x+1] == v {
					uf.connect_ids_atomic(idx_xy, idx_up + 1);
				}
            }
        }
    }
}

fn do_unionfind_line2b(uf: &mut impl UnionFind<(u32, u32), Id = u32>, im: &ImageY8, y: usize) {
    assert!(y > 0);
	// debug_assert_eq!(im.width(), uf.width as usize);
	// #[cfg(debug_assertions)]
	// debug_assert_eq!(im.height(), uf.height as usize);

    let mut v_0_m1 = im[(0, y-1)];
    let mut v_1_m1 = im[(1, y-1)];
    let mut v = im[(0,y)];

	let w = im.width();

	let row = im.row(y);
	let row_up = im.row(y-1);

	for x in 1..(w - 1) {
        let v_m1_m1 = v_0_m1;
        v_0_m1 = v_1_m1;
        v_1_m1 = im[(x + 1, y - 1)];
        let v_m1_0 = v;
        v = row[x];

        if v == 127 {
			continue;
		}

		let idx_xy = uf.index_to_id((x as _, y as _));
		let idx_up = uf.index_to_id((x as _, y as u32 - 1));

        // (dx,dy) pairs for 8 connectivity:
        // (-1, -1)    (0, -1)    (1, -1)
        // (-1, 0)    (REFERENCE)
		if row[x-1] == v {
			uf.connect_ids(idx_xy, idx_xy - 1);
		}

        if x == 1 || !((v_m1_0 == v_m1_m1) && (v_m1_m1 == v_0_m1)) {
			if row_up[x] == v {
				uf.connect_ids(idx_xy, idx_up);
			}
        }

        if v == 255 {
            if x == 1 || !(v_m1_0 == v_m1_m1 || v_0_m1 == v_m1_m1) {
				if row_up[x-1] == v {
					uf.connect_ids(idx_xy, idx_up - 1);
				}
            }
            if v_0_m1 != v_1_m1 {
				if row_up[x+1] == v {
					uf.connect_ids(idx_xy, idx_up + 1);
				}
            }
        }
    }
}

fn do_unionfind_line2c(uf: &mut impl UnionFind<(u32, u32), Id = u32>, im: &ImageY8, y: usize) {
    assert!(y > 0);
	// debug_assert_eq!(im.width(), uf.width as usize);
	// #[cfg(debug_assertions)]
	// debug_assert_eq!(im.height(), uf.height as usize);

    let mut v_0_m1 = im[(0, y-1)];
    let mut v_1_m1 = im[(1, y-1)];
    let mut v = im[(0,y)];

	let w = im.width();

	let row = im.row(y);
	let row_up = im.row(y-1);

	for x in 1..(w - 1) {
        let v_m1_m1 = v_0_m1;
        v_0_m1 = v_1_m1;
        v_1_m1 = im[(x + 1, y - 1)];
        let v_m1_0 = v;
        v = row[x];

        if v == 127 {
			continue;
		}

		let idx_xy = uf.index_to_id((x as _, y as _));
		let idx_up = uf.index_to_id((x as _, y as u32 - 1));

        // (dx,dy) pairs for 8 connectivity:
        // (-1, -1)    (0, -1)    (1, -1)
        // (-1, 0)    (REFERENCE)
		if row[x-1] == v {
			uf.connect_ids(idx_xy, idx_xy - 1);
		}

        if x == 1 || !((v_m1_0 == v_m1_m1) && (v_m1_m1 == v_0_m1)) {
			if row_up[x] == v {
				uf.connect_ids(idx_xy, idx_up);
			}
        }

        if v == 255 {
            if x == 1 || !(v_m1_0 == v_m1_m1 || v_0_m1 == v_m1_m1) {
				if row_up[x-1] == v {
					uf.connect_ids(idx_xy, idx_up - 1);
				}
            }
            if v_0_m1 != v_1_m1 {
				if row_up[x+1] == v {
					uf.connect_ids(idx_xy, idx_up + 1);
				}
            }
        }
    }
}

fn do_unionfind_first_line(uf: &mut impl UnionFind<(u32, u32)>, im: &ImageY8) {
	for x in 1..(im.width()-1) {
		let v0 = im[(x, 0)];
		if v0 == 127 {
			continue;
		}
		let v1 = im[(x - 1, 0)];
		if v0 == v1 {
			uf.connect((x as _, 0), (x as u32 - 1, 0));
		}
	}
}

pub(crate) fn connected_components(config: &DetectorConfig, threshim: &ImageY8) -> impl UnionFindStatic<(u32, u32), Id = u32> {
    if config.single_thread() {
        let mut uf = UnionFind2D::new(threshim.width(), threshim.height());
	    do_unionfind_first_line(&mut uf, threshim);
		for y in 1..threshim.height() {
            do_unionfind_line2b(&mut uf, threshim, y);
        }
        uf
    } else {
        let mut uf = UnionFind2D::new_concurrent(threshim.width(), threshim.height());
	    do_unionfind_first_line(&mut uf, threshim);
		let height = threshim.height();
		let chunksize = 1 + height / (config.nthreads().get() * 2);
        // each task will process [y0, y1). Note that this attaches
        // each cell to the right and down, so row y1 *is* potentially modified.
        //
        // for parallelization, make sure that each task doesn't touch rows
        // used by another thread.
        (1..height)
            .into_par_iter()
            .step_by(chunksize)
            .for_each(|i| {
                let y0 = i;
                let y1 = std::cmp::min(i + chunksize, height);
                for y in y0..y1 {
                    do_unionfind_line2(&uf, threshim, y);
                }
            });
        
        UnionFind2D::from_concurrent(uf)
    }
}