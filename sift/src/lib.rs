pub mod downsample_x2;
pub mod gaussian_blur;
pub mod sift_image;

use gaussian_blur::{GaussianBlurGpu, GaussianKernelCpu, GaussianKernelGpu};
use image::DynamicImage;
use lazy_static::lazy_static;
use std::path::Path;

use crate::gaussian_blur::GaussianBlurCpu;
use crate::sift_image::Image;

// https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

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

const CONTRAST_THRESHOLD: i16 = (0.03 * 255.0) as i16; // threshold for contrast check in extrema detection

const MAX_INTERPOLATION_ATTEMPTS: usize = 5;

lazy_static! {
    // constant multiplicative factor K for the blur in an octave
    static ref K: f64 = 2_f64.powf(1.0 / S as f64);
}

pub struct SiftKeyPoint {
    octave: usize,
    layer: usize,
    pub x: usize,
    pub y: usize,
}

pub struct SiftCpu {
    octaves: Option<usize>,
    blurs: Vec<Vec<Box<GaussianBlurCpu>>>,
}

impl SiftCpu {
    pub fn new() -> SiftCpu {
        SiftCpu {
            octaves: None,
            blurs: vec![],
        }
    }

    /// Knowing the image size in advance gives a performance boost
    pub fn new_fixed_size(image_width: usize, image_height: usize) -> SiftCpu {
        let octaves = SiftCpu::calc_num_octaves(image_width, image_height);
        let blurs = SiftCpu::init_gaussian_blur_pyramid_cpu(octaves);
        SiftCpu {
            octaves: Some(octaves),
            blurs,
        }
    }

    pub fn run(&mut self, image: &DynamicImage) -> Vec<SiftKeyPoint> {
        let width = image.width() as usize;
        let height = image.height() as usize;
        let raw_grayscale_image = image
            .grayscale()
            .into_luma8()
            .iter()
            .map(|x| *x as i16)
            .collect();
        let image = Image::from_raw(raw_grayscale_image, width, height);

        let g_pyramid = self.compute_scale_space_cpu(&image);

        for octave in 0..g_pyramid.len() {
            for layer in 0..g_pyramid[octave].len() {
                g_pyramid[octave][layer]
                    .save_grayscale(Path::new(&format!("gaussian_{}_{}.png", octave, layer)));
            }

            // downsample_x2(&mut g_pyramid[octave][0]);
            // g_pyramid[octave][0].save_grayscale(Path::new(&format!("downsampled_{}.png", octave)));
        }

        let dog_pyramid = Self::compute_dog_pyramid_cpu(g_pyramid);

        for octave in 0..dog_pyramid.len() {
            for layer in 0..dog_pyramid[octave].len() {
                dog_pyramid[octave][layer]
                    .save_grayscale(Path::new(&format!("dog_{}_{}.png", octave, layer)));
            }
        }

        let keypoints = Self::find_extrema_cpu(dog_pyramid);

        return keypoints;
    }

    /// Calculate the number of octaves we will be able to
    /// use based on the size of the image
    fn calc_num_octaves(width: usize, height: usize) -> usize {
        let mut image_size = std::cmp::min(width, height) / 2;
        let mut radius = 0;
        let mut octave = 1;
        while radius < image_size {
            let stdev = octave as f64 * SIGMA * K.powi((N_G - 1) as i32);
            radius = GaussianKernelCpu::radius(stdev) as usize;
            octave += 1;
            image_size /= 2;
        }
        return octave - 1;
    }

    /// Instantiate the gaussian blur operations that will be needed to build the pyramid.
    /// We do this separately so we have the option of generating the kernels ahead of time
    /// if we know the image dimensions.
    fn init_gaussian_blur_pyramid_cpu(octaves: usize) -> Vec<Vec<Box<GaussianBlurCpu>>> {
        let mut blurs: Vec<Vec<Box<GaussianBlurCpu>>> = vec![];
        for i in 0..octaves {
            blurs.push(vec![]);
            let base_sigma = (i + 1) as f64 * SIGMA; // scale the base std dev for each octave
            let sigma_0 = 0.5f64 * base_sigma;
            blurs[i].push(Box::new(GaussianBlurCpu::new(sigma_0)));
            for j in 1..N_G {
                let sigma = K.powi((j - 1).try_into().unwrap()) * base_sigma;
                println!("octave: {}, layer: {}, sigma: {}", i, j, sigma);
                blurs[i].push(Box::new(GaussianBlurCpu::new(sigma)));
            }
        }
        return blurs;
    }

    /// Compute the Gaussian scale-space representation of an image
    /// Assumes image is grayscale
    fn compute_scale_space_cpu(&mut self, image: &Image) -> Vec<Vec<Image>> {
        let octaves = self
            .octaves
            .unwrap_or_else(|| SiftCpu::calc_num_octaves(image.width, image.height));

        if self.blurs.is_empty() {
            self.blurs = Self::init_gaussian_blur_pyramid_cpu(octaves);
        }

        let mut gaussians_pyramid: Vec<Vec<Image>> = vec![];
        gaussians_pyramid.reserve_exact(octaves);

        // generate octaves
        let mut scaled = image.clone();
        for octave in 0..octaves {
            gaussians_pyramid.push(vec![]);
            gaussians_pyramid[octave].reserve_exact(N_G);
            for g in 0..N_G {
                self.blurs[octave][g].apply(&mut scaled);
                // scaled.save_grayscale(Path::new(&format!("pyramid_{}_{}.png", i, j)));
                gaussians_pyramid[octave].push(scaled.clone());
            }
            scaled.downsample_x2();
        }
        return gaussians_pyramid;
    }

    fn compute_dog_pyramid_cpu(gaussians_pyramid: Vec<Vec<Image>>) -> Vec<Vec<Image>> {
        // generate DoG pyramid
        let mut dogs_pyramid: Vec<Vec<Image>> = vec![];
        dogs_pyramid.reserve_exact(gaussians_pyramid.len());
        for octave in 0..gaussians_pyramid.len() {
            dogs_pyramid.push(vec![]);
            dogs_pyramid[octave].reserve_exact(N_DOG);
            for layer in 1..N_G {
                let mut diff = gaussians_pyramid[octave][layer - 1].clone();
                diff -= &gaussians_pyramid[octave][layer];
                dogs_pyramid[octave].push(diff);
            }
        }
        return dogs_pyramid;
    }

    fn find_extrema_cpu(dog_pyramid: Vec<Vec<Image>>) -> Vec<SiftKeyPoint> {
        let mut keypoints: Vec<SiftKeyPoint> = vec![];
        // find scale-space minima and maxima at each level
        for octave in 0..dog_pyramid.len() - 2 {
            for layer in 1..N_DOG - 1 {
                let before = &dog_pyramid[octave][layer - 1];
                let after = &dog_pyramid[octave][layer + 1];
                let image = &dog_pyramid[octave][layer];
                for row in 1..image.height - 1 {
                    // leave a 1 pixel border
                    for column in 1..image.width - 1 {
                        let i = row * image.width + column;
                        let value = image.data[i];
                        let is_extrema = (value >= CONTRAST_THRESHOLD
                            && Self::is_maxima(value, i, &before, false)
                            && Self::is_maxima(value, i, &image, true)
                            && Self::is_maxima(value, i, &after, false))
                            || (value <= -CONTRAST_THRESHOLD
                                && Self::is_minima(value, i, &before, false)
                                && Self::is_minima(value, i, &image, true)
                                && Self::is_minima(value, i, &after, false));
                        if is_extrema {
                            let mut keypoint = SiftKeyPoint {
                                octave,
                                layer,
                                x: column,
                                y: row,
                            };

                            if !Self::refine_extrema_cpu(&dog_pyramid, &mut keypoint) {
                                // bad keypoint
                                continue;
                            }

                            keypoints.push(keypoint);
                        }
                    }
                }
            }
        }
        return keypoints;
    }

    /// No bounds checking performed.
    /// Assumes that 0 < x < width-1 and 0 < y < height-1
    fn is_maxima(value: i16, i: usize, image: &Image, skip_i: bool) -> bool {
        return value > image.data[i - image.width - 1] && // top left
            value > image.data[i - image.width] &&
            value > image.data[i - image.width + 1] &&
            value > image.data[i - 1] &&
            (skip_i || (value > image.data[i])) &&
            value > image.data[i + 1] &&
            value > image.data[i + image.width - 1] &&
            value > image.data[i + image.width] &&
            value > image.data[i + image.width + 1]; // bottom left
    }

    /// No bounds checking performed.
    /// Assumes that 0 < x < width-1 and 0 < y < height-1
    fn is_minima(value: i16, i: usize, image: &Image, skip_i: bool) -> bool {
        return value < image.data[i - image.width - 1] && // top left
            value < image.data[i - image.width] &&
            value < image.data[i - image.width + 1] &&
            value < image.data[i - 1] &&
            (skip_i || (value < image.data[i])) &&
            value < image.data[i + 1] &&
            value < image.data[i + image.width - 1] &&
            value < image.data[i + image.width] &&
            value < image.data[i + image.width + 1]; // bottom left
    }

    fn refine_extrema_cpu(dog_pyramid: &Vec<Vec<Image>>, keypoint: &mut SiftKeyPoint) -> bool {
        // interpolate to find the sub-pixel and sub-scale accurate keypoint locations

        return true;
    }
}

pub struct SiftGpu<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    octaves: usize,
    blurs: Vec<Vec<Box<GaussianBlurGpu<'a>>>>,
}

impl<'a> SiftGpu<'a> {
    /// This is the implementation of the SIFT algorithm where as much as possible is done on the GPU
    /// Knowing the image size in advance gives a performance boost
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        image_width: usize,
        image_height: usize,
    ) -> SiftGpu<'a> {
        let octaves = SiftGpu::calc_num_octaves(image_width, image_height);
        let blurs = SiftGpu::init_gaussian_blur_pyramid_gpu(&device, &queue, octaves);
        SiftGpu {
            device,
            queue,
            octaves,
            blurs,
        }
    }

    pub fn run(&mut self, image: &DynamicImage) {
        let width = image.width() as usize;
        let height = image.height() as usize;
        let raw_grayscale_image = image
            .grayscale()
            .into_luma8()
            .iter()
            .map(|x| *x as i16)
            .collect();
        let image = Image::from_raw(raw_grayscale_image, width, height);

        self.compute_scale_space_gpu(&image);
    }

    /// Calculate the number of octaves we will be able to
    /// use based on the size of the image
    fn calc_num_octaves(width: usize, height: usize) -> usize {
        let mut image_size = std::cmp::min(width, height) / 2;
        let mut radius = 0;
        let mut octave = 1;
        while radius < image_size {
            let stdev = octave as f64 * SIGMA * K.powi((N_G - 1) as i32);
            radius = GaussianKernelGpu::radius(stdev as f32) as usize;
            octave += 1;
            image_size /= 2;
        }

        return octave - 1;
    }

    /// Instantiate the gaussian blur operations that will be needed to build the pyramid.
    /// We do this separately so we have the option of generating the kernels ahead of time
    /// if we know the image dimensions.
    fn init_gaussian_blur_pyramid_gpu(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        octaves: usize,
    ) -> Vec<Vec<Box<GaussianBlurGpu<'a>>>> {
        let mut blurs: Vec<Vec<Box<GaussianBlurGpu>>> = vec![];
        for i in 0..octaves {
            blurs.push(vec![]);
            let base_sigma = (i + 1) as f64 * SIGMA; // scale the base std dev for each octave
            for j in 0..N_G {
                let sigma = K.powi(j.try_into().unwrap()) * base_sigma;
                blurs[i].push(Box::new(GaussianBlurGpu::new(
                    &device,
                    &queue,
                    sigma as f32,
                )));
            }
        }
        return blurs;
    }

    /// Compute the Gaussian scale-space representation of an image
    /// Assumes image is grayscale
    fn compute_scale_space_gpu(&mut self, image: &Image) {
        let mut gaussians_pyramid: Vec<Vec<Image>> = vec![];
        gaussians_pyramid.reserve_exact(self.octaves);

        // generate octaves
        let mut scaled = image.clone();
        for i in 0..self.octaves {
            gaussians_pyramid.push(vec![]);
            for j in 0..N_G {
                let blur = &mut self.blurs[i][j];
                blur.apply(&mut scaled);
                // scaled.save_grayscale(Path::new(&format!("pyramid_{}_{}.png", i, j)));
                gaussians_pyramid[i].push(scaled.clone());
            }
            scaled.downsample_x2();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use image::DynamicImage;

    use crate::{gaussian_blur::GaussianBlurGpu, sift_image::Image};

    pub fn blur_gpu(image: DynamicImage) {
        let width = image.width() as usize;
        let height = image.height() as usize;
        let raw_grayscale_image = image
            .grayscale()
            .into_luma8()
            .iter()
            .map(|x| *x as i16)
            .collect();
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

    #[test]
    fn test_blur_gpu() {
        let img_bytes = include_bytes!("../test_data/harbourside.png");
        let img = image::load_from_memory(img_bytes).unwrap();
        blur_gpu(img);
    }
}
