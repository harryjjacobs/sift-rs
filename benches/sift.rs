use divan::Bencher;

use sift::{
    gaussian_blur::{GaussianBlurCpu, GaussianBlurGpu},
    sift_image::Image,
    SiftCpu, SiftGpu,
};

fn main() {
    // Run `add` benchmark:
    divan::main();
}

fn image_from_bytes(bytes: &[u8]) -> Image {
    let image = image::load_from_memory(bytes).unwrap().grayscale();
    return Image::from_raw(
        image.as_luma8().unwrap().clone().into_raw(),
        image.width() as usize,
        image.height() as usize,
    );
}

fn setup_wgpu() -> (wgpu::Device, wgpu::Queue) {
    // make wgpu logging visible
    let _ = env_logger::try_init();

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

    return pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None, // Trace path
    ))
    .unwrap();
}

#[divan::bench()]
fn benchmark_gaussian_blur_cpu(bencher: Bencher) {
    let blur = GaussianBlurCpu::new(1.6);
    let image_bytes = include_bytes!("../test_data/harbourside.png");
    let mut image = image_from_bytes(image_bytes);

    bencher.bench_local(move || {
        blur.apply(&mut image);
    });
}

#[divan::bench()]
fn benchmark_gaussian_blur_gpu(bencher: Bencher) {
    let (device, queue) = setup_wgpu();
    let mut blur = GaussianBlurGpu::new(&device, &queue, 1.6);
    let image_bytes = include_bytes!("../test_data/harbourside.png");
    let mut image = image_from_bytes(image_bytes);

    bencher.bench_local(move || {
        blur.apply(&mut image);
    });
}

#[divan::bench()]
fn benchmark_sift_cpu_sized(bencher: Bencher) {
    let img_bytes = include_bytes!("../test_data/harbourside.png");
    let image = image::load_from_memory(img_bytes).unwrap();

    let mut sift = SiftCpu::new_fixed_size(image.width() as usize, image.height() as usize);
    bencher.bench_local(move || {
        sift.run(&image);
    });
}

#[divan::bench()]
fn benchmark_sift_gpu(bencher: Bencher) {
    let img_bytes = include_bytes!("../test_data/harbourside.png");
    let image = image::load_from_memory(img_bytes).unwrap();

    let (device, queue) = setup_wgpu();

    let mut sift = SiftGpu::new(
        &device,
        &queue,
        image.width() as usize,
        image.height() as usize,
    );

    bencher.bench_local(move || {
        sift.run(&image);
    });
}
