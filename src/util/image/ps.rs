use std::io::{self, Write, Result};

use crate::util::{image::Pixel, geom::Point2D};

use super::Rgb;

pub trait ImageWritePostscript {
	/// Write PostScript data
	fn write_postscript(&self, f: &mut PostScriptWriter<impl io::Write>) -> io::Result<()>;
}

pub struct PostScriptWriter<'a, W: io::Write> {
    inner: &'a mut W,
}

impl<'a, W: Write> Write for PostScriptWriter<'a, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

impl<'a, W: io::Write> PostScriptWriter<'a, W> {
    pub fn new(inner: &'a mut W) -> Result<Self> {
        write!(inner, "%%!PS\n\n")?;
        Ok(Self { inner })
    }

    pub fn scale(&mut self, scale_x: f32, scale_y: f32) -> Result<()> {
        writeln!(self.inner, "{:.15} {:.15} scale", scale_x, scale_y)
    }

    pub fn translate(&mut self, dx: f32, dy: f32) -> Result<()> {
        writeln!(self.inner, "{} {} translate", dx, dy)
    }

    pub fn setrgbcolor(&mut self, rgb: &Rgb<u8>) -> Result<()> {
        writeln!(self.inner, "{} {} {} setrgbcolor",
            rgb.channels()[0] as f64/255.,
            rgb.channels()[1] as f64/255.,
            rgb.channels()[2] as f64/255.
        )
    }

    pub(crate) fn path(&mut self, callback: impl FnOnce(&mut PSCommandWriter<W>) -> Result<()>) -> Result<()> {
        {
            let mut child = PSCommandWriter(self.inner);
            callback(&mut child)?;
        }
        writeln!(self.inner)
    }

    pub fn showpage(&mut self) -> Result<()> {
        writeln!(self.inner, "showpage")
    }
}

pub(crate) trait VectorPathWriter {
    fn move_to(&mut self, point: &Point2D) -> Result<()>;
    fn line_to(&mut self, point: &Point2D) -> Result<()>;
}

pub (crate) struct PSCommandWriter<'a, W: Write>(&'a mut W);

impl<'a, W: Write> VectorPathWriter for PSCommandWriter<'a, W> {
    fn move_to(&mut self, point: &Point2D) -> Result<()> {
        write!(self.0, "{} {} moveto ", point.x(), point.y())
    }

    fn line_to(&mut self, point: &Point2D) -> Result<()> {
        write!(self.0, "{} {} lineto ", point.x(), point.y())
    }
}

impl<'a, W: Write> PSCommandWriter<'a, W> {
    pub fn stroke(&mut self) -> Result<()> {
        write!(self.0, "stroke ")
    }
}