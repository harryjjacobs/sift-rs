use image::GenericImageView;

use sift::{blur_gpu, sift_cpu};

fn main() {
    println!("Hello, world!");

    let img_bytes = include_bytes!("../test_data/harbourside.png");
    let img = image::load_from_memory(img_bytes).unwrap();
    sift_cpu(img);

    // blur_gpu(img);
}
