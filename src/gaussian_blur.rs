use core::panic;

use crate::sift_image::Image;
use lazy_static::lazy_static;
use wgpu::{util::DeviceExt, BufferBinding};

lazy_static! {
    static ref SQRT2PI_F64: f64 = (2.0 * std::f64::consts::PI).sqrt();
    static ref SQRT2PI_F32: f32 = (2.0 * std::f32::consts::PI).sqrt();
}

pub fn gaussian_distribution_f32(stdev: f32, mean: f32, x: f32) -> f32 {
    return (1.0 / (stdev * *SQRT2PI_F32))
        * std::f32::consts::E.powf(-0.5 * ((x - mean) * (x - mean)) / (stdev * stdev));
}

pub struct GaussianKernelCpu {
    pub data: Vec<f64>,
    pub radius: i32,
}

impl GaussianKernelCpu {
    pub fn new(stdev: f64) -> GaussianKernelCpu {
        let radius = Self::radius(stdev) as i32;
        let mut data = Vec::new();
        for i in -radius..=radius {
            data.push(GaussianKernelCpu::gaussian_distribution_f64(
                stdev, 0.0, i as f64,
            ));
        }
        let sum: f64 = data.iter().sum();
        for i in 0..data.len() {
            data[i] /= sum;
        }
        return GaussianKernelCpu { data, radius };
    }

    pub fn gaussian_distribution_f64(stdev: f64, mean: f64, x: f64) -> f64 {
        return (1.0 / (stdev * *SQRT2PI_F64))
            * (-0.5 * ((x - mean) * (x - mean)) / (stdev * stdev)).exp();
    }

    pub fn radius(stdev: f64) -> u32 {
        return (3.0 * stdev).ceil() as u32;
    }
}

pub struct GaussianBlurCpu {
    kernel: GaussianKernelCpu,
}

impl GaussianBlurCpu {
    pub fn new(stdev: f64) -> GaussianBlurCpu {
        return GaussianBlurCpu {
            kernel: GaussianKernelCpu::new(stdev),
        };
    }

    pub fn apply(&self, image: &mut Image) {
        let kernel_radius = self.kernel.radius;
        let kernel = &self.kernel.data;

        if kernel.len() as usize >= image.width || kernel.len() as usize >= image.height {
            panic!("Kernel size is bigger than the image. Image width: {}, height: {}. Kernel size: {}", 
                image.width,
                image.height,
                kernel.len(),
            );
        }

        let width = image.width;
        let height = image.height;
        let original_image = image.clone();
        // blur along x
        let x_max = width as i32 - 1;
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let mut sum = 0.0;
                for i in 0..kernel_radius * 2 + 1 {
                    // reflect
                    let mut ii = x - kernel_radius + i;
                    if ii < 0 {
                        ii = ii.abs(); // reflect around 0
                    } else if ii > x_max {
                        ii = x_max - (ii - x_max); // reflect around max x
                    }
                    let pixel = original_image.at(ii as usize, y as usize);
                    let pixel_value = pixel as f64;
                    sum += pixel_value * kernel[i as usize];
                }
                *image.at_mut(x as usize, y as usize) = sum;
            }
        }
        let original_image = image.clone();
        // blur along y
        let y_max = height as i32 - 1;
        for x in 0..width as i32 {
            for y in 0..height as i32 {
                let mut sum = 0.0;
                for i in 0..kernel_radius * 2 + 1 {
                    // reflect
                    let mut ii = y - kernel_radius + i;
                    if ii < 0 {
                        ii = ii.abs(); // reflect around 0
                    } else if ii > y_max {
                        ii = y_max - (ii - y_max); // reflect around max x
                    }
                    let pixel = original_image.at(x as usize, ii as usize);
                    let pixel_value = pixel as f64;
                    sum += pixel_value * kernel[i as usize];
                }
                *image.at_mut(x as usize, y as usize) = sum;
            }
        }
    }
}
