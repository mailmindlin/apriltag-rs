use std::num::NonZeroU32;

use crate::util::ImageY8;

#[derive(PartialEq, Eq)]
enum PixelType {
	Nothing,
	White,
	Black,
	Data,
	Recursive,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PixelTypeFromCharError {
	InvalidCharacter(char),
}

impl TryFrom<char> for PixelType {
	type Error = PixelTypeFromCharError;

	fn try_from(value: char) -> Result<Self, Self::Error> {
		match value {
			'x' => Ok(Self::Nothing),
			'd' => Ok(Self::Data),
			'w' => Ok(Self::Data),
			'b' => Ok(Self::Black),
			'r' => panic!("Not implemented"),
			_ => Err(PixelTypeFromCharError::InvalidCharacter(value)),
		}
	}
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ImageLayoutError {
	/// Zero size
	Empty,
	/// Dimensions are not square
	NotSquare,
	/// Invalid character
	InvalidCharacter(usize, PixelTypeFromCharError),
	/// Pattern is not symmetric
	NotSymmetric,
	/// Pattern has no border
	NoBorder,
	SizeOverflow,
}


pub struct ImageLayout {
	name: String,
	num_bits: usize,
	size: NonZeroU32,
	border_width: u32,
	border_start: u32,
	reversed_border: bool,

	pixels: Vec<Vec<PixelType>>,
}

const fn l1DistToEdge(x: u32, y: u32, size: u32) -> u32 {
	let res = size - 1 - y;
	let res = if x < res { x } else { res };
	let r2 = size - 1 - x;
	let res = if r2 < res { r2 } else { res };
	if y < res { y } else { res }
}

fn l2DistToCenter(x: u32, y: u32, size: u32) -> f64 {
	let r = size as f64 / 2.;
	
	f64::hypot(x as f64 + 0.5 - r, y as f64 + 0.5 - r)
}

fn validateBorder(layout: &ImageLayout, loc: usize, reversed: bool) -> bool {
	let (outside, inside) = if reversed {
		(PixelType::Black, PixelType::White)
	} else {
		(PixelType::White, PixelType::Black)
	};

	for x in loc..(layout.size.get() as usize - loc - 1) {
		if layout.pixels[loc][x] != outside {
			return false;
		}
	}

	for x in (loc + 1)..(layout.size.get() as usize - loc - 2) {
		if layout.pixels[loc + 1][x] != inside {
			return false;
		}
	}

	true
}

impl ImageLayout {
	pub fn for_name(specifier: String) -> Result<Self, ImageLayoutError> {
		if specifier.starts_with("classic_") {
			let size = u32::from_str_radix(&specifier[8..], 10).unwrap();
			Self::new_classic(size)
        } else if specifier.starts_with("standard_") {
			let size = u32::from_str_radix(&specifier[9..], 10).unwrap();
			Self::new_standard(size)
        } else if specifier.starts_with("circle_") {
			let size = u32::from_str_radix(&specifier[7..], 10).unwrap();
			Self::new_circle(size)
        } else if specifier.starts_with("custom_") {
			Self::from_string("Custom".into(), &specifier[7..])
        } else {
			panic!("Invalid layout specification.")
        }
	}
	pub fn new_classic(size: u32) -> Result<Self, ImageLayoutError> {
		if size == 0 {
			return Err(ImageLayoutError::Empty);
		}

		let cap = (size as usize).checked_mul(size as _).ok_or(ImageLayoutError::SizeOverflow)?;
		let mut pattern = Vec::with_capacity(cap);
		for y in 0..size {
			for x in 0..size {
				let value = match l1DistToEdge(x, y, size) {
					0 => PixelType::White,
					1 => PixelType::Black,
					_ => PixelType::Data,
				};
				pattern.push(value);
			}
		}
		// Classic layout has no name for backwards compatibility.
		Self::from_pattern("".into(), pattern)
	}

	pub fn new_standard(size: u32) -> Result<Self, ImageLayoutError> {
		if size == 0 {
			return Err(ImageLayoutError::Empty);
		}

		let cap = (size as usize).checked_mul(size as _).ok_or(ImageLayoutError::SizeOverflow)?;
		let mut pattern = Vec::with_capacity(cap);
		for y in 0..size {
			for x in 0..size {
				let pixel = match l1DistToEdge(x, y, size) {
					1 => PixelType::Black,
					2 => PixelType::White,
					_ => PixelType::Data,
				};
				pattern.push(pixel);
			}
		}
		Self::from_pattern("Standard".into(), pattern)
	}

	pub fn new_circle(size: u32) -> Result<Self, ImageLayoutError> {
		if size == 0 {
			return Err(ImageLayoutError::Empty);
		}

		let cap = (size as usize).checked_mul(size as _).ok_or(ImageLayoutError::SizeOverflow)?;
		let mut pattern = Vec::with_capacity(cap);

		let cutoff = (size as f64 / 2.) - 0.25;
		let border_distance = (size as f64/2. - cutoff*f64::sqrt(0.5) - 0.5).ceil() as u32;
		for y in 0..size {
			for x in 0..size {
				let dist = l1DistToEdge(x, y, size);
				let pixel = if dist == border_distance {
					PixelType::Black
				} else if dist == border_distance + 1 {
					PixelType::White
				} else if l2DistToCenter(x, y, size) <= cutoff {
					PixelType::Data
				} else {
					PixelType::Nothing
				};
				pattern.push(pixel);
			}
		}

		Self::from_pattern("Circle".into(), pattern)
	}

	fn from_pattern(name: String, pattern: Vec<PixelType>) -> Result<Self, ImageLayoutError> {
		let size = f64::sqrt(pattern.len() as _) as u32;
		if size.saturating_mul(size) as usize != pattern.len() {
			return Err(ImageLayoutError::NotSquare);
		}

		let num_bits = pattern.iter()
			.filter(|p| *p == &PixelType::Data)
			.count();

		let mut layout = Self {
			name,
			num_bits,
			size: NonZeroU32::new(size).ok_or(ImageLayoutError::Empty)?,
			border_width: 0, // Will be filled in later
			border_start: 0, // Will be filled in later
			reversed_border: false,
			pixels: vec![],
		};

		// Validate symmetric under rot90
		for y in 0..=(size as usize / 2) {
			for x in y..(size as usize - 1 - y) {
				if !(layout.pixels[y][x] == layout.pixels[x][layout.size() as usize - 1 - y]
						&& layout.pixels[y][x] == layout.pixels[layout.size() as usize - 1 - x][y]
						&& layout.pixels[y][x] == layout.pixels[layout.size() as usize - 1 - y][layout.size() as usize - 1 - x]) {
					return Err(ImageLayoutError::NotSymmetric);
				}
			}
		}

		// Validate border
		let mut found_border = false;
		for i in 0..((size as usize - 1) / 2) {
			if layout.pixels[i][i] == PixelType::White
					&& layout.pixels[i + 1][i + 1] == PixelType::Black
					&& validateBorder(&layout, i, false) {
				found_border = true;
				layout.reversed_border = false;
				layout.border_start = i as u32 + 1;
				layout.border_width = layout.size.get() - 2*layout.border_start;
				break;
			}
			if layout.pixels[i][i] == PixelType::Black
					&& layout.pixels[i + 1][i + 1] == PixelType::White
					&& validateBorder(&layout, i, true) {
				found_border = true;
				layout.reversed_border = true;
				layout.border_start = i as u32 + 1;
				layout.border_width = layout.size.get() - 2*layout.border_start;
				break;
			}
		}
		if !found_border {
			return Err(ImageLayoutError::NoBorder);
		}

		Ok(layout)
	}

	pub fn from_string(name: String, data: &str) -> Result<Self, ImageLayoutError> {
		let pattern = data.chars()
			.enumerate()
			.map(|(idx, c)| {
				PixelType::try_from(c)
					.map_err(|e| ImageLayoutError::InvalidCharacter(idx, e))	
			})
			.collect::<Result<Vec<_>, _>>()?;
		Self::from_pattern(name, pattern)
	}

	pub const fn num_bits(&self) -> usize {
		self.num_bits
	}

	pub fn render_to_image(&self, code: u64) -> ImageY8 {
		let image_data = self.render_to_array(code);

		let size = self.size.get() as usize;
		let mut im = ImageY8::zeroed_packed(size, size);
		for y in 0..size {
			for x in 0..size {
				let value = match image_data[y][x] {
					0 => 0,
					1 => 255,
					2 => 128, //TODO: transparent
					_ => panic!("Unknown image pixel color"),
				};
				im[(x, y)] = value;
			}
		}
		im
	}

	/// Render to an int array. Used for rendering image output and also for computing complexity.
	pub fn render_to_array(&self, mut code: u64) -> Vec<Vec<i32>> {
		let size = self.size.get() as usize;
		let mut im = vec![vec![0; size]; size];

		fn rotate90<T: Default + Clone>(im1: Vec<Vec<T>>) -> Vec<Vec<T>> {
			let size = im1.len();
			let mut res = vec![vec![T::default(); size]; size];
	
			for y in 0..size {
				for x in 0..size {
					res[size - 1 - x][y].clone_from(&im1[y][x]);
				}
			}
			res
		}

		for i in 0..4 {
			im = rotate90(im);
			// Render one-quarter of the image
			for y in 0..=(size / 2) {
				for x in y..(size - 1 - y) {
					let color = match self.pixels[y as usize][x as usize] {
						PixelType::Data => {
							let color = if (code & (1 << (self.num_bits - 1))) != 0 {
								1
							} else {
								0
							};

							code <<= 1;
							color
						},
						PixelType::Black => 0,
						PixelType::White => 1,
						PixelType::Nothing => 2,
						_ => unreachable!(),
					};
					im[y as usize][x as usize] = color;
				}
			}
		}

		// If there is a middle pixel, set it.
		if size % 2 == 1 {
			let middle = size as usize / 2;
			im[middle][middle] = match self.pixels[middle][middle] {
				PixelType::Data =>
					if code & (1 << self.num_bits - 1) != 0 {
						1
					} else {
						0
					}
				PixelType::Black => 0,
				PixelType::White => 1,
				PixelType::Nothing => 2,
				_ => panic!("Impossible state"),
			};
		}
		// Rotate back to 0º
		rotate90(im)
	}

	pub const fn size(&self) -> u32 {
		self.size.get()
	}

	fn is_reversed_border(&self) -> bool {
		self.reversed_border
	}

	pub fn bit_locations(&self) -> Vec<(u32, u32)> {
		let mut locations = Vec::with_capacity(self.num_bits);
		let size = self.size.get();
		for y in 0..(size/2) {
			for x in y..(size - 1 - y) {
				if self.pixels[y as usize][x as usize] == PixelType::Data {
					locations.push((x as _, y as _));
				}
			}
		}

		let step = self.num_bits / 4;
		while locations.len() < step * 4 {
			let idx = locations.len();
			let (px, py) = &locations[locations.len() - (step as usize)];
			locations.push((
				size - 1 - *py,
				*px
			));
		}

		// Middle pixel.
		if locations.len() < self.num_bits as _ {
			locations.push((size / 2, size / 2));
		}


		// Shift the origin.
		for (x, y) in locations.iter_mut() {
			*x -= self.border_start;
			*y -= self.border_start;
		}

		locations
	}

	pub fn border_width(&self) -> u32 {
		self.border_width
	}

	pub fn name(&self) -> &str {
		&self.name
	}
}