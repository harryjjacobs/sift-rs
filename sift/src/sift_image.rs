use std::path::Path;

use image::GrayImage;

#[derive(Clone)]
pub struct Image {
    pub data: Vec<i16>,
    pub width: usize,
    pub height: usize,
}

impl Image {
    pub fn from_raw(data: Vec<i16>, width: usize, height: usize) -> Image {
        Image {
            data,
            width,
            height,
        }
    }

    pub fn at(&self, x: usize, y: usize) -> i16 {
        return self.data[x + self.width * y];
    }

    pub fn at_mut(&mut self, x: usize, y: usize) -> &mut i16 {
        return &mut self.data[x + self.width * y];
    }

    pub fn save_grayscale(&self, path: &Path) {
        let image = GrayImage::from_raw(
            self.width as u32,
            self.height as u32,
            self.data.iter().map(|x| (*x).clamp(0, 255) as u8).collect(),
        )
        .unwrap();
        let _ = image.save(path);
    }
}

fn sub_assign(lhs: &mut Image, rhs: &Image) {
    if lhs.height != rhs.height || lhs.width != rhs.width {
        panic!("Images must be the same size");
    }
    let size = lhs.width * lhs.height;
    for i in 0..size {
        lhs.data[i] = rhs.data[i].saturating_sub(lhs.data[i]);
    }
}

impl std::ops::SubAssign<Image> for Image {
    fn sub_assign(&mut self, rhs: Image) {
        sub_assign(self, &rhs);
    }
}

impl std::ops::SubAssign<&Image> for Image {
    fn sub_assign(&mut self, rhs: &Image) {
        sub_assign(self, &rhs);
    }
}

impl std::ops::SubAssign<Image> for &mut Image {
    fn sub_assign(&mut self, rhs: Image) {
        sub_assign(self, &rhs);
    }
}

impl std::ops::SubAssign<&Image> for &mut Image {
    fn sub_assign(&mut self, rhs: &Image) {
        sub_assign(self, &rhs);
    }
}
