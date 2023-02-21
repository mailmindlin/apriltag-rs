use std::{path::Path, io};

use super::{Image, PNM, pnm::PNMFormat, ImageWritePNM};

pub type ImageRGB = Image<[u8; 3]>;

impl Image<[u8; 3]> {
    /// Least common multiple of 64 (sandy bridge cache line) and 48 (stride needed
    /// for 16byte-wide RGB processing). (It's possible that 48 would be enough).
    const DEFAULT_ALIGNMENT: usize = 192;
    pub fn create(width: usize, height: usize) -> Self {
        Self::create_alignment(width, height, Self::DEFAULT_ALIGNMENT)
    }

    // Create an RGB image from PNM
    pub fn create_from_pnm(path: &Path) -> io::Result<Self> {
        let pnm = PNM::create_from_file(path)?;
        match pnm.format {
            PNMFormat::Binary => todo!("Support binary files"),
            PNMFormat::Gray => {
                let im = Self::create(pnm.width, pnm.height);

                for y in 0..im.height {
                    for x in 0..im.width {
                        let gray = pnm.buf[y*im.width + x];
                        im[(x, y)] = [gray; 3];
                    }
                }
                Ok(im)
            },
            PNMFormat::RGB => {
                let im = Self::create(pnm.width, pnm.height);

                for y in 0..im.height {
                    for x in 0..im.width {
                        let r = pnm.buf[y*im.width*3 + 3*x];
                        let g = pnm.buf[y*im.width*3 + 3*x+1];
                        let b = pnm.buf[y*im.width*3 + 3*x+2];

                        im[(x, y)] = [r, g, b];
                    }
                }
                Ok(im)
            },
        }
    }

    
}

impl ImageWritePNM for Image<[u8; 3]> {
    fn write_pnm(&self, f: &mut impl io::Write) -> io::Result<()> {
        // Only outputs to RGB
        writeln!(f, "P6")?;
        writeln!(f, "{} {}", self.width, self.height)?;
        writeln!(f, "255")?;
        let linesz = self.width * 3;
        for y in 0..self.height {
            let row = &self[(.., y)];
            f.write_all(row.flatten())?;
        }

        Ok(())
    }
}