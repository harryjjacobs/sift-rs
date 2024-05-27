use std::path::Path;

use image::GrayImage;

#[derive(Clone)]
pub struct Image {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

impl Image {
    pub fn from_raw(data: Vec<u8>, width: usize, height: usize) -> Image {
        Image {
            data,
            width,
            height,
        }
    }

    pub fn at(&self, x: usize, y: usize) -> u8 {
        return self.data[x + self.width * y];
    }

    pub fn at_mut(&mut self, x: usize, y: usize) -> &mut u8 {
        return &mut self.data[x + self.width * y];
    }

    pub fn save_grayscale(&self, path: &Path) {
        let image =
            GrayImage::from_raw(self.width as u32, self.height as u32, self.data.clone()).unwrap();
        let _ = image.save(path);
    }
}
