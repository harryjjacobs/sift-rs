pub mod gaussian_blur;
pub mod sift_image;

use gaussian_blur::GaussianBlurGpu;
use image::DynamicImage;
use std::path::Path;

use crate::gaussian_blur::GaussianBlurCpu;
use crate::sift_image::Image;

// https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

const DEFAULT_OCTAVES: usize = 4;

const SIGMA: f64 = 1.6;

const S: usize = 3; // the number of scales within each octave

/// Pairs of blurred images make a DoG image.
/// We need to compare with neighbouring scales above and below (left and right) (in the same octave)
/// (e.g. to have 3 DoG images all each have neighbours j - 1 and j + 1, there need to be 5 of them)
const N_DOG: usize = S + 2; // the number of difference-of-gaussians images in each octave

/// "We must produce s + 3 images in the stack of blurred images for each octave,
/// so that final extrema detection covers a complete octave." See above comment
/// on N_DOG
const N_G: usize = S + 3; // the number of gaussian-blurred images in each octave

/// Compute the Gaussian scale-space representation of an image
/// Assumes image is grayscale
fn compute_scale_space_cpu<const OCTAVES: usize>(image: &Image) {
    let sigma: f64 = SIGMA;
    let k = 2_f64.powf(1.0 / S as f64); // factor k
    let gaussians_pyramid: [[Option<Box<DynamicImage>>; N_G]; OCTAVES] =
        std::array::from_fn(|_| std::array::from_fn(|_| None));
    let mut scaled = image.clone();
    // generate octaves
    for i in 0..OCTAVES {
        let base_sigma = (i + 1) as f64 * sigma; // scale the base std dev for each octave
        for j in 0..N_G {
            let sigma = k.powi(j.try_into().unwrap()) * base_sigma;
            println!("sigma: {}", sigma);
            let blur = GaussianBlurCpu::new(sigma);
            blur.apply(&mut scaled);
            scaled.save_grayscale(Path::new("tmp.png"));
            // return;
        }
        // // downsample by a factor of two
        // scaled.resize_exact(
        //     scaled.width() / 2_u32.pow(i as u32 + 1),
        //     scaled.height() / 2_u32.pow(i as u32 + 1),
        //     FilterType::Nearest,
        // );
    }
}

pub fn sift_cpu(image: DynamicImage) {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let raw_grayscale_image = image.grayscale().as_luma8().unwrap().clone().into_raw();
    let image = Image::from_raw(raw_grayscale_image, width, height);
    compute_scale_space_cpu::<DEFAULT_OCTAVES>(&image);
}

pub fn blur_gpu(image: DynamicImage) {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let raw_grayscale_image = image.grayscale().as_luma8().unwrap().clone().into_raw();
    let mut image = Image::from_raw(raw_grayscale_image, width, height);

    // make wgpu logging visible
    env_logger::init();

    // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        #[cfg(not(target_arch = "wasm32"))]
        backends: wgpu::Backends::PRIMARY,
        #[cfg(target_arch = "wasm32")]
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });

    // Handle to the graphics card
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .unwrap();

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None, // Trace path
    ))
    .unwrap();

    let mut blur = GaussianBlurGpu::new(&device, &queue, 1.6);
    blur.apply(&mut image);

    image.save_grayscale(Path::new("tmp_gpu.png"));
}

#[cfg(test)]
mod tests {
    // use super::*;
}
