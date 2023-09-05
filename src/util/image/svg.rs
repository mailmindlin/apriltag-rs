use std::io::{self, Write, Result};

use crate::util::{image::Pixel, geom::Point2D};

use super::{Rgb, VectorPathWriter};

pub trait ImageWriteSVG {
	/// Write SVG data
	fn write_svg(&self, f: &mut SVGWriter<impl Write>) -> io::Result<()>;
}

pub struct SVGWriter<'a, W: Write> {
    inner: &'a mut W,
}

impl<'a, W: Write> Write for SVGWriter<'a, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

impl<'a, W: Write> SVGWriter<'a, W> {
    pub fn new(inner: &'a mut W) -> Result<Self> {
        write!(inner, "<svg version=\"1.1\" width=\"300\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">")?;
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

    pub(crate) fn path(&mut self, callback: impl FnOnce(&mut SVGPathWriter<W>) -> Result<()>) -> Result<()> {
        {
            let mut child = SVGPathWriter::new(self.inner)?;
            callback(&mut child)?;
        }
        writeln!(self.inner)
    }

    pub(crate) fn finish(self) -> Result<()> {
        write!(self.inner, "</svg>")?;
        Ok(())
    }
}

pub (crate) struct SVGPathWriter<'a, W: Write> {
    inner: &'a mut W,
    stroke: Option<Rgb<u8>>,
    fill: Option<Rgb<u8>>,
}

impl<'a, W: Write> VectorPathWriter for SVGPathWriter<'a, W> {
    fn move_to(&mut self, point: &Point2D) -> Result<()> {
        write!(self.inner, "M {} {} ", point.x(), point.y())
    }

    fn line_to(&mut self, point: &Point2D) -> Result<()> {
        write!(self.inner, "L {} {} ", point.x(), point.y())
    }
}

impl<'a, W: Write> SVGPathWriter<'a, W> {
    fn new(inner: &'a mut W) -> Result<Self> {
        write!(inner, "<path d=\"")?;
        Ok(Self {
            inner,
            stroke: None,
            fill: None,
        })
    }
    pub fn stroke(&mut self) -> Result<()> {
        write!(self.inner, "stroke ")
    }
}