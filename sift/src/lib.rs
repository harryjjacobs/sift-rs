pub mod downsample_x2;
pub mod gaussian_blur;
pub mod sift_image;

use gaussian_blur::{GaussianBlurGpu, GaussianKernelCpu, GaussianKernelGpu};
use image::DynamicImage;
use lazy_static::lazy_static;
use std::f64::consts::PI;
use std::path::Path;

use crate::gaussian_blur::GaussianBlurCpu;
use crate::sift_image::Image;

// https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

const SIGMA: f64 = 1.6;
const SIGMA0: f64 = 0.5; // initial blur level of the input image

const S: usize = 3; // the number of scales within each octave

/// Pairs of blurred images make a DoG image.
/// We need to compare with neighbouring scales above and below (left and right) (in the same octave)
/// (e.g. to have 3 DoG images all each have neighbours j - 1 and j + 1, there need to be 5 of them)
const N_DOG: usize = S + 2; // the number of difference-of-gaussians images in each octave

/// "We must produce s + 3 images in the stack of blurred images for each octave,
/// so that final extrema detection covers a complete octave." See above comment
/// on N_DOG.
/// We produce S + 3 gaussian blurred images, so that we can produce S + 2 DoG images.
/// The 5 DoG images allow for the comparison of pixels in every possible pair of adjacent scale levels.
/// This setup ensures that keypoints can be detected at the beginning, middle, and end of the scale
/// range, providing complete coverage.
const N_G: usize = S + 3; // the number of gaussian-blurred images in each octave

const CONTRAST_THRESHOLD: f64 = 0.03 * 255.0; // threshold for contrast check in extrema detection

const EDGE_THRESHOLD_R: f64 = 10.0; // threshold for edge response elimination in extrema detection

const MAX_INTERPOLATION_ATTEMPTS: usize = 5; // max number of iterations when refining extrema location

const ORIENTATION_HISTOGRAM_BINS: usize = 36; // number of bins in the orientation histogram (10 deg resolution)

// the factor to apply to the sigma of the gaussian window used to weight the samples added to the histogram.
// The sigma value used for this gaussian-weighted circular window should be this factor times the keypoint
// scale.
const ORIENTATION_SIGMA_FACTOR: f64 = 1.5;

const ORIENTATION_PEAK_RATIO: f64 = 0.8; // ratio of the peak value to other local peaks in the orientation
                                         // histogram to decide whether an additional keypoint to be created

const DESCRIPTOR_WINDOW_WIDTH: usize = 16; // the width of the descriptor window in pixels
const DESCRIPTOR_WIDTH: usize = 4; // the width of the descriptor in pixels

lazy_static! {
    // constant multiplicative factor K for the blur in an octave
    static ref K: f64 = 2_f64.powf(1.0 / S as f64);
}

#[derive(Debug, Clone)]
pub struct SiftKeyPoint {
    pub octave: usize,
    pub layer: usize,

    // the coordinates normalised to the original image
    // (not scaled for the layer)
    pub x: f64,
    pub y: f64,
    pub scale: f64,

    pub layer_x: f64,
    pub layer_y: f64,
    pub layer_scale: f64,

    pub orientation: f64,
    pub magnitude: f64,

    // The 128-element descriptor vector.
    // There are 8 orientation bins for each of the 4x4 descriptor regions
    pub descriptor: [f64; DESCRIPTOR_WIDTH * DESCRIPTOR_WIDTH * 8],
}

pub struct SiftCpu {
    octaves: Option<usize>,
    blurs: Vec<Box<GaussianBlurCpu>>,
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
        let blurs = SiftCpu::init_gaussian_blur_pyramid_cpu();
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
            .map(|x| *x as f64)
            .collect();
        let image = Image::from_raw(raw_grayscale_image, width, height);

        let g_pyramid = self.compute_scale_space_cpu(&image);
        let dog_pyramid = Self::compute_dog_pyramid_cpu(&g_pyramid);
        let (grad_mag_pyramid, grad_ori_pyramid) =
            Self::compute_gradient_magnitudes_orientations_cpu(&g_pyramid);

        for octave in 0..dog_pyramid.len() {
            for layer in 0..dog_pyramid[octave].len() {
                dog_pyramid[octave][layer]
                    .save_grayscale(Path::new(&format!("dog_{}_{}.png", octave, layer)));
            }
        }

        let mut keypoints =
            Self::detect_keypoints(dog_pyramid, &grad_mag_pyramid, &grad_ori_pyramid);
        println!("Detected {} keypoints", keypoints.len());

        Self::extract_descriptors(&grad_mag_pyramid, &grad_ori_pyramid, &mut keypoints);

        return keypoints;
    }

    /// Calculate the number of octaves we will be able to
    /// use based on the size of the image
    fn calc_num_octaves(width: usize, height: usize) -> usize {
        let mut image_size = std::cmp::min(width, height) / 2;
        let mut octave = 1;
        let max_stdev = SIGMA * K.powi((N_G - 1) as i32);
        let max_radius = GaussianKernelCpu::radius(max_stdev) as usize;
        while image_size > max_radius {
            octave += 1;
            image_size /= 2;
        }
        return octave;
    }

    /// Instantiate the gaussian blur operations that will be needed to build the pyramid.
    /// We do this separately so we have the option of generating the kernels ahead of time.
    fn init_gaussian_blur_pyramid_cpu() -> Vec<Box<GaussianBlurCpu>> {
        let mut blurs: Vec<Box<GaussianBlurCpu>> = vec![];
        // Because we apply the gaussian blur cumulatively (we blur the previous, already
        // blurred image), we need to calculate the relative sigma for each blur.
        blurs.push(Box::new(GaussianBlurCpu::new(
            (SIGMA.powi(2) - SIGMA0.powi(2)).sqrt(),
        )));
        for j in 1..N_G {
            // the sigma of the blur in the previous layer
            let sigma_previous = K.powi((j - 1).try_into().unwrap()) * SIGMA;
            // the total sigma that we want to achieve with this blur
            let sigma_total = sigma_previous * *K;
            // println!("sigma_total: {}", sigma_total);
            // the increment in sigma that we need to achieve with this blur
            // i.e. sigma_total^2 = sigma_previous^2 + sigma_increment^2
            // Note to self: you can derive this by calculating the convolution of two gaussians.
            // The variance term in the resulting gaussian is the sum of the sigmas of the two input gaussians squared.
            let sigma_increment = (sigma_total.powi(2) - sigma_previous.powi(2)).sqrt();
            blurs.push(Box::new(GaussianBlurCpu::new(sigma_increment)));
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
            self.blurs = Self::init_gaussian_blur_pyramid_cpu();
        }

        let mut gaussians_pyramid: Vec<Vec<Image>> = vec![];
        gaussians_pyramid.reserve_exact(octaves);

        // generate octaves
        let mut scaled = image.clone();
        for octave in 0..octaves {
            gaussians_pyramid.push(vec![]);
            gaussians_pyramid[octave].reserve_exact(N_G);
            for g in 0..N_G {
                if g > 0 || octave == 0 {
                    // don't blur the first layer (unless this is the first octave)
                    self.blurs[g].apply(&mut scaled);
                }
                gaussians_pyramid[octave].push(scaled.clone());
                scaled.save_grayscale(Path::new(&format!("pyramid_{}_{}.png", octave, g)));
            }
            // "Once a complete octave has been processed, we resample the Gaussian image
            // that has twice the initial value of σ (it will be 2 images from the top of
            // the stack) by taking every second pixel in each row and column."
            scaled = gaussians_pyramid[octave][S].clone();
            scaled.downsample_x2();
        }
        return gaussians_pyramid;
    }

    /// Compute the gradient magnitudes and orientations for each pixel in each image in the pyramid
    fn compute_gradient_magnitudes_orientations_cpu(
        gaussians_pyramid: &Vec<Vec<Image>>,
    ) -> (Vec<Vec<Image>>, Vec<Vec<Image>>) {
        let mut magnitudes_pyramid: Vec<Vec<Image>> = vec![];
        let mut orientations_pyramid: Vec<Vec<Image>> = vec![];
        for octave in 0..gaussians_pyramid.len() {
            magnitudes_pyramid.push(vec![]);
            orientations_pyramid.push(vec![]);
            for layer in 1..=S {
                // 1..S+1 in your code; Equivalent: 1..=S
                let image = &gaussians_pyramid[octave][layer];
                let mut mags = Image::new_zero(image.width, image.height);
                let mut oris = Image::new_zero(image.width, image.height);
                for y in 1..image.height - 1 {
                    for x in 1..image.width - 1 {
                        let dx = image.at(x + 1, y) - image.at(x - 1, y);
                        let dy = image.at(x, y + 1) - image.at(x, y - 1);
                        let mag = (dx * dx + dy * dy).sqrt();
                        let mut ang = dy.atan2(dx);
                        if ang < 0.0 {
                            // atan2 returns values in the range -PI to PI.
                            ang += 2.0 * PI;
                        }
                        *mags.at_mut(x, y) = mag;
                        *oris.at_mut(x, y) = ang;
                    }
                }
                magnitudes_pyramid[octave].push(mags);
                orientations_pyramid[octave].push(oris);
            }
        }
        (magnitudes_pyramid, orientations_pyramid)
    }

    fn compute_dog_pyramid_cpu(gaussians_pyramid: &Vec<Vec<Image>>) -> Vec<Vec<Image>> {
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

    fn detect_keypoints(
        dog_pyramid: Vec<Vec<Image>>,
        grad_mag_pyramid: &Vec<Vec<Image>>,
        grad_ori_pyramid: &Vec<Vec<Image>>,
    ) -> Vec<SiftKeyPoint> {
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
                                layer_x: column as f64,
                                layer_y: row as f64,
                                layer_scale: 0.0,
                                scale: 0.0,
                                x: 0.0,
                                y: 0.0,
                                magnitude: 0.0,
                                orientation: 0.0,
                                descriptor: [0f64; DESCRIPTOR_WIDTH * DESCRIPTOR_WIDTH * 8],
                            };

                            if !Self::refine_extrema_cpu(&dog_pyramid, &mut keypoint) {
                                // bad keypoint
                                continue;
                            }

                            Self::compute_gradient_histogram(
                                &grad_mag_pyramid[keypoint.octave][keypoint.layer - 1],
                                &grad_ori_pyramid[keypoint.octave][keypoint.layer - 1],
                                &mut keypoint,
                                &mut keypoints,
                            );

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
    fn is_maxima(value: f64, i: usize, image: &Image, skip_i: bool) -> bool {
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
    fn is_minima(value: f64, i: usize, image: &Image, skip_i: bool) -> bool {
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

    fn refine_extrema_cpu(dog_pyramid: &Vec<Vec<Image>>, kp: &mut SiftKeyPoint) -> bool {
        // Interpolate to find the sub-pixel and sub-scale accurate keypoint locations.

        // We take the taylor expansion of the Difference-of-gaussian function, at the
        // given keypoint position (x, y, sigma)

        // change in DoG function wrt x
        let before = &dog_pyramid[kp.octave][kp.layer - 1];
        let current = &dog_pyramid[kp.octave][kp.layer];
        let after = &dog_pyramid[kp.octave][kp.layer + 1];

        let mut x = kp.layer_x as usize;
        let mut y = kp.layer_y as usize;

        let layer = kp.layer;

        let mut dx;
        let mut dy;
        let mut ds;
        let mut dxx;
        let mut dyy;
        let mut dxy;

        let mut offset: [f64; 3] = [0f64; 3];

        let mut refined_kp_x = 0f64;
        let mut refined_kp_y = 0f64;
        let mut refined_kp_layer = 0f64;

        let mut i = 0usize;
        loop {
            if i >= MAX_INTERPOLATION_ATTEMPTS {
                return false;
            }

            if x == 0
                || x >= (current.width as i32 - 1) as usize
                || y == 0
                || y >= (current.height as i32 - 1) as usize
            {
                // the keypoint is too close to the edge
                return false;
            }

            // https://en.wikipedia.org/wiki/Finite_difference#Multivariate_finite_differences

            // First  derivatives
            dx = (current.at(x + 1, y) - current.at(x - 1, y)) * 0.5;
            dy = (current.at(x, y + 1) - current.at(x, y - 1)) * 0.5;
            ds = (after.at(x, y) - before.at(x, y)) * 0.5;

            // Second  derivatives
            dxx = current.at(x + 1, y) - 2.0 * current.at(x, y) + current.at(x - 1, y);
            dyy = current.at(x, y + 1) - 2.0 * current.at(x, y) + current.at(x, y - 1);
            let dss = after.at(x, y) - 2.0 * current.at(x, y) + before.at(x, y);

            // Cross derivatives
            dxy = (current.at(x + 1, y + 1) - current.at(x + 1, y - 1) - current.at(x - 1, y + 1)
                + current.at(x - 1, y - 1))
                * 0.25;
            let dxs = (after.at(x + 1, y) - before.at(x + 1, y) - after.at(x - 1, y)
                + before.at(x - 1, y))
                * 0.25;
            let dys = (after.at(x, y + 1) - before.at(x, y + 1) - after.at(x, y - 1)
                + before.at(x, y - 1))
                * 0.25;

            /*
                ∇D = dx dy ds

                     dxx dxy dxs
                H  = dyx dyy dys
                     dsx dsy dss

                x = -(HD^-1)(∇D)
            */

            // The hessian (2nd order partial derivative)
            let h = [[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]];

            // Inverse of the hessian

            // https://en.wikipedia.org/wiki/Leibniz_formula_for_determinants
            let det = h[0][0] * h[1][1] * h[2][2]
                + h[0][1] * h[1][2] * h[2][0]
                + h[0][2] * h[1][0] * h[2][1]
                - h[0][2] * h[1][1] * h[2][0]
                - h[0][1] * h[1][0] * h[2][2]
                - h[0][0] * h[1][2] * h[2][1];

            if det.abs() < f64::MIN {
                // det == 0, not invertible
                break;
            }

            let mut h_inv = [[0f64; 3]; 3];

            // Compute the inverse using the adjugate matrix divided by the determinant
            let det_reciprocal = 1.0 / det;
            h_inv[0][0] = det_reciprocal * (h[1][1] * h[2][2] - h[1][2] * h[2][1]);
            h_inv[1][0] = det_reciprocal * (h[1][2] * h[2][0] - h[1][0] * h[2][2]);
            h_inv[2][0] = det_reciprocal * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);
            h_inv[0][1] = det_reciprocal * (h[0][2] * h[2][1] - h[0][1] * h[2][2]);
            h_inv[1][1] = det_reciprocal * (h[0][0] * h[2][2] - h[0][2] * h[2][0]);
            h_inv[2][1] = det_reciprocal * (h[0][1] * h[2][0] - h[0][0] * h[2][1]);
            h_inv[0][2] = det_reciprocal * (h[0][1] * h[1][2] - h[0][2] * h[1][1]);
            h_inv[1][2] = det_reciprocal * (h[0][2] * h[1][0] - h[0][0] * h[1][2]);
            h_inv[2][2] = det_reciprocal * (h[0][0] * h[1][1] - h[0][1] * h[1][0]);

            // Multiply the hessian with the gradient and negate to get the offset
            offset[0] = -(h_inv[0][0] * dx + h_inv[0][1] * dy + h_inv[0][2] * ds);
            offset[1] = -(h_inv[1][0] * dx + h_inv[1][1] * dy + h_inv[1][2] * ds);
            offset[2] = -(h_inv[2][0] * dx + h_inv[2][1] * dy + h_inv[2][2] * ds);

            refined_kp_x = x as f64 + offset[0];
            refined_kp_y = y as f64 + offset[1];
            refined_kp_layer = layer as f64 + offset[2];

            if offset[0] > 0.5 && (x as i32) < (current.width as i32 - 2) {
                x += 1;
            } else if offset[0] < -0.5 && x > 0 {
                x -= 1;
            } else if offset[1] > 0.5 && (y as i32) < (current.height as i32 - 2) {
                y += 1;
            } else if offset[1] < -0.5 && y > 0 {
                y -= 1;
            } else {
                // the feature is not closer to an adjacent pixel, we are refined
                break;
            }

            i += 1;
        }

        if refined_kp_x < 0.0
            || refined_kp_x >= current.width as f64
            || refined_kp_y < 0.0
            || refined_kp_y >= current.height as f64
            || refined_kp_layer < 0.0
            || refined_kp_layer >= dog_pyramid[kp.octave].len() as f64
        {
            // don't continue if the refined position is out of range
            return false;
        }

        // D(x, y, s) + 0.5 * offset * gradient
        let interpolated_value =
            current.at(x, y) as f64 + 0.5 * (dx * offset[0] + dy * offset[1] + ds * offset[2]);

        if interpolated_value.abs() < CONTRAST_THRESHOLD {
            // contrast is below the threshold, the refined keypoint is not an extrema. discard
            return false;
        }

        // edge response elimination
        let tr = dxx + dyy;
        let det_sp = dxx * dyy - dxy * dxy;
        if det_sp <= 0.0 {
            return false;
        }
        let r = EDGE_THRESHOLD_R;
        let edge_ratio = (tr * tr) / det_sp;
        if edge_ratio >= ((r + 1.0) * (r + 1.0) / r) {
            return false;
        }

        kp.layer_x = refined_kp_x;
        kp.layer_y = refined_kp_y;
        kp.layer = layer; // just stay in the same layer? Is this right?

        // the scale of the keypoint is the sigma of the layer in which it was found
        kp.layer_scale = SIGMA * K.powf(refined_kp_layer);

        // calculate normalised coordinates and scale for the original image
        // i.e. octave 3 is 8 times smaller than the original image
        let octave_factor = 2f64.powf(kp.octave as f64);
        kp.scale = kp.layer_scale * octave_factor;
        kp.x = kp.layer_x * octave_factor;
        kp.y = kp.layer_y * octave_factor;

        return true;
    }

    fn interpolate_hist_bin(prev: f64, current: f64, next: f64) -> f64 {
        // interpolate the peak position by fitting a parabola to the three histogram values.
        // i.e. the peak isn't necessarily at the bin with the highest value but could be between two bins.
        // we can derive the formula for the parabola by solving the system of equations for the three points:
        // (0, prev), (1, current), (2, next), AX^2 + BX + C = Y
        // the resulting quadratic function will have a maximum at the peak which we can find by taking the derivative.
        let denom = next + prev - 2.0 * current;
        if denom.abs() < f64::MIN {
            return 0.0;
        }
        return 0.5 * (prev - next) / denom;
    }

    fn compute_gradient_histogram(
        gradient_magnitudes: &Image,
        gradient_orientations: &Image,
        keypoint: &mut SiftKeyPoint,
        keypoints: &mut Vec<SiftKeyPoint>,
    ) {
        let mut hist = [0f64; ORIENTATION_HISTOGRAM_BINS];

        let sigma = ORIENTATION_SIGMA_FACTOR * keypoint.layer_scale;
        let radius = GaussianKernelCpu::radius(sigma) as i32;

        let kp_x = keypoint.layer_x;
        let kp_y = keypoint.layer_y;

        let bin_width = 2.0 * PI / ORIENTATION_HISTOGRAM_BINS as f64;

        for j in -radius..=radius {
            for i in -radius..=radius {
                let x = (kp_x + i as f64).round() as isize;
                let y = (kp_y + j as f64).round() as isize;

                if x < 1
                    || y < 1
                    || x >= gradient_magnitudes.width as isize - 1
                    || y >= gradient_magnitudes.height as isize - 1
                {
                    continue;
                }

                let magnitude = gradient_magnitudes.at(x as usize, y as usize);
                let mut orientation = gradient_orientations.at(x as usize, y as usize);

                // map to [0, 2π)
                if orientation < 0.0 {
                    orientation += 2.0 * PI;
                }
                if orientation >= 2.0 * PI {
                    orientation -= 2.0 * PI;
                }

                // We only need to compute the exponential component of the gaussian as the normalisation constant
                // will be common to all magnitudes.
                // 2D gaussian
                let radius_squared =
                    (x as f64 - kp_x) * (x as f64 - kp_x) + (y as f64 - kp_y) * (y as f64 - kp_y);
                let weight = (-radius_squared / (2.0 * sigma * sigma)).exp();

                let weighted_magnitude = weight * magnitude;

                // the angle will actually fall somewhere between two bins so we should
                // interpolate the value between the two bins.
                let bin = orientation / bin_width;
                let b0 = bin.floor() as isize;
                let frac = bin - b0 as f64; // how much to interpolate to the next bin
                let b1 = b0 + 1;

                hist[(b0.rem_euclid(ORIENTATION_HISTOGRAM_BINS as isize)) as usize] +=
                    weighted_magnitude * (1.0 - frac);
                hist[(b1.rem_euclid(ORIENTATION_HISTOGRAM_BINS as isize)) as usize] +=
                    weighted_magnitude * frac;
            }
        }

        // Find the biggest bin
        let mut biggest = 0.0;
        let mut max_bin = 0;
        for i in 0..ORIENTATION_HISTOGRAM_BINS {
            if hist[i] > biggest {
                biggest = hist[i];
                max_bin = i;
            }
        }

        // set the orientation of the original keypoint to the highest peak
        keypoint.orientation = max_bin as f64 * bin_width;

        // for neighbouring bins that are within 80% of the biggest bin, create additional keypoints.
        // this improves the stability of the keypoints
        let local_peak_threshold = ORIENTATION_PEAK_RATIO * biggest;
        for bin in 0..ORIENTATION_HISTOGRAM_BINS {
            let next = hist[(bin + 1) % ORIENTATION_HISTOGRAM_BINS];
            let current = hist[bin];
            let prev;
            if bin == 0 {
                prev = *hist.last().unwrap();
            } else {
                prev = hist[bin - 1];
            }

            // create additional keypoints for any local peaks within 80% of the highest peak
            if current > prev && current > next && current >= local_peak_threshold {
                let offset = Self::interpolate_hist_bin(prev, current, next); // in range -0.5..0.5
                let interpolated_bin =
                    (bin as f64 + offset).clamp(0.0, ORIENTATION_HISTOGRAM_BINS as f64);

                let mut new_keypoint = keypoint.clone();
                new_keypoint.orientation = interpolated_bin * bin_width;
                if new_keypoint.orientation < 0.0 {
                    new_keypoint.orientation += 2.0 * PI;
                }
                if new_keypoint.orientation >= 2.0 * PI {
                    new_keypoint.orientation -= 2.0 * PI;
                }
                keypoints.push(new_keypoint);
            }
        }
    }

    fn extract_descriptors(
        grad_mag_pyramid: &Vec<Vec<Image>>,
        grad_ori_pyramid: &Vec<Vec<Image>>,
        keypoints: &mut Vec<SiftKeyPoint>,
    ) {
        let desc_bins = 8usize; // 8 orientation bins per cell
        let cells = DESCRIPTOR_WIDTH; // 4x4 cells
        let window_width = DESCRIPTOR_WINDOW_WIDTH as f64; // 16
        let half_window_width = window_width / 2.0;

        // Gaussian weighting (σ = half the window)
        let sigma = 0.5 * window_width;
        let invsig2 = 1.0 / (2.0 * sigma * sigma);

        for kp in keypoints.iter_mut() {
            // Work in the keypoint's layer coordinates
            let mags = &grad_mag_pyramid[kp.octave][kp.layer - 1];
            let oris = &grad_ori_pyramid[kp.octave][kp.layer - 1];

            let cos_t = kp.orientation.cos();
            let sin_t = kp.orientation.sin();

            // each descriptor element accumulates with trilinear interpolation:
            // (row cell, col cell, orientation bin)
            let mut desc = [0.0; 128];

            // sample points over the 16x16 window around the keypoint.
            // we sample at pixel centers in layer space.
            let step = 1.0;
            let start = -half_window_width + 0.5;
            let end = half_window_width - 0.5;

            // imagine a grid of points over the image, then rotate that grid to the keypoint orientation,
            // then iterate over each of the points in the grid to sample the image gradient magnitude and orientation
            // and accumulate into the descriptor with trilinear interpolation.
            for dy in (0..)
                .map(|k| start + k as f64 * step)
                .take_while(|&v| v <= end)
            {
                for dx in (0..)
                    .map(|k| start + k as f64 * step)
                    .take_while(|&v| v <= end)
                {
                    // rotate (dx,dy) to the keypoint frame, then scale by keypoint.layer_scale
                    let rx = (cos_t * dx + sin_t * dy) / kp.layer_scale;
                    let ry = (-sin_t * dx + cos_t * dy) / kp.layer_scale;

                    // position in image (layer) coords
                    let x = kp.layer_x + dx;
                    let y = kp.layer_y + dy;

                    if x <= 1.0
                        || y <= 1.0
                        || x >= (mags.width as f64 - 2.0)
                        || y >= (mags.height as f64 - 2.0)
                    {
                        continue;
                    }

                    // gaussian window weight
                    let w = (-(dx * dx + dy * dy) * invsig2).exp();

                    // gradient orientation and magnitude at the closest pixel
                    let xi = x.round() as usize;
                    let yi = y.round() as usize;
                    let mut orientation = oris.at(xi, yi) - kp.orientation;
                    while orientation < 0.0 {
                        orientation += 2.0 * PI;
                    }
                    while orientation >= 2.0 * PI {
                        orientation -= 2.0 * PI;
                    }
                    let mag = mags.at(xi, yi) * w;

                    // map the rotated coordinates to descriptor cell coordinates [0,cells]
                    let cell_x = (rx / (window_width / cells as f64)) + (cells as f64 / 2.0);
                    let cell_y = (ry / (window_width / cells as f64)) + (cells as f64 / 2.0);
                    if cell_x < 0.0
                        || cell_x >= cells as f64
                        || cell_y < 0.0
                        || cell_y >= cells as f64
                    {
                        continue;
                    }

                    // map orientation to [0, desc_bins]
                    // these are the 8 descriptor orientation bins per cell
                    let orientation_bin: f64 = orientation * (desc_bins as f64) / 2.0 * PI;

                    // we perform trilinear interpolation among 8 neighbours (row,col,orient)
                    let r0 = cell_y.floor(); // integer part of the cell coordinates
                    let c0 = cell_x.floor(); // integer part of the cell coordinates
                    let o0 = orientation_bin.floor(); // integer part of the orientation bin

                    let dr = cell_y - r0; // fractional part of the cell coordinates
                    let dc = cell_x - c0; // fractional part of the cell coordinates
                    let do_ = orientation_bin - o0; // fractional part of the orientation bin

                    let r0u = r0 as usize;
                    let c0u = c0 as usize;
                    let o0u = o0 as usize % desc_bins;

                    // each of the 8 neighbours receives a weighted contribution
                    // (1-dx)(1-dy)(1-do), (1-dx)(1-dy)(do), (1-dx)(dy)(1-do), (1-dx)(dy)(do),
                    // (dx)(1-dy)(1-do), (dx)(1-dy)(do), (dx)(dy)(1-do), (dx)(dy)(do)
                    for (rr, wr) in [(r0u, 1.0 - dr), (r0u + 1, dr)] {
                        if rr >= cells {
                            continue;
                        }
                        for (cc, wc) in [(c0u, 1.0 - dc), (c0u + 1, dc)] {
                            if cc >= cells {
                                continue;
                            }
                            let wrc = wr * wc;

                            let o1 = (o0u + 1) % desc_bins;
                            let w0 = (1.0 - do_) * wrc * mag;
                            let w1 = do_ * wrc * mag;

                            let base = (rr * cells + cc) * desc_bins;
                            desc[base + o0u] += w0;
                            desc[base + o1] += w1;
                        }
                    }
                }
            }

            // normalize to unit length
            let mut norm = 0.0;
            for v in &desc {
                norm += v * v;
            }
            norm = norm.sqrt();
            if norm > 1e-12 {
                for v in &mut desc {
                    *v /= norm;
                }
            }

            // clamp at 0.2 then renormalize
            for v in &mut desc {
                if *v > 0.2 {
                    *v = 0.2;
                }
            }
            let mut norm2 = 0.0;
            for v in &desc {
                norm2 += v * v;
            }
            norm2 = norm2.sqrt();
            if norm2 > 1e-12 {
                for v in &mut desc {
                    *v /= norm2;
                }
            }

            // Store
            for i in 0..128 {
                kp.descriptor[i] = desc[i];
            }
        }
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
            .map(|x| *x as f64)
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
            gaussians_pyramid[i].reserve_exact(N_G);
            for j in 0..N_G {
                self.blurs[i][j].apply(&mut scaled);
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
            .map(|x| *x as f64)
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

    #[test]
    fn test_interpolate_hist_bin() {
        let prev = 1.0;
        let current = 4.0;
        let next = 5.0;
        let interpolated = super::SiftCpu::interpolate_hist_bin(prev, current, next);
        assert_eq!(interpolated, 1.0);
    }
}
