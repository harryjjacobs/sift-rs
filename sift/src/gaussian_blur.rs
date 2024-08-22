use core::panic;

use crate::sift_image::Image;
use lazy_static::lazy_static;
use wgpu::{util::DeviceExt, BufferBinding};

lazy_static! {
    static ref SQRT2PI_F64: f64 = (2.0 * std::f64::consts::PI).sqrt();
    static ref SQRT2PI_F32: f32 = (2.0 * std::f32::consts::PI).sqrt();
}

fn gaussian_distribution_f64(stdev: f64, mean: f64, x: f64) -> f64 {
    return (1.0 / (stdev * *SQRT2PI_F64))
        * std::f64::consts::E.powf(-0.5 * ((x - mean) * (x - mean)) / (stdev * stdev));
}

fn gaussian_distribution_f32(stdev: f32, mean: f32, x: f32) -> f32 {
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
            data.push(gaussian_distribution_f64(stdev, 0.0, i as f64));
        }
        let sum: f64 = data.iter().sum();
        for i in 0..data.len() {
            data[i] /= sum;
        }
        return GaussianKernelCpu { data, radius };
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
                *image.at_mut(x as usize, y as usize) = sum as i16;
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
                *image.at_mut(x as usize, y as usize) = sum as i16;
            }
        }
    }
}

pub struct GaussianKernelGpu {
    pub data: Vec<f32>,
    pub radius: i32,
}

impl GaussianKernelGpu {
    pub fn new(stdev: f32) -> GaussianKernelGpu {
        let radius = Self::radius(stdev);
        let mut data = Vec::new();
        for i in -radius..=radius {
            data.push(gaussian_distribution_f32(stdev, 0.0, i as f32));
        }
        let sum: f32 = data.iter().sum();
        for i in 0..data.len() {
            data[i] /= sum;
        }
        return GaussianKernelGpu { data, radius };
    }

    pub fn radius(stdev: f32) -> i32 {
        return (3.0 * stdev).ceil() as i32;
    }
}

pub struct GaussianBlurGpu<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    compute_pipeline: wgpu::ComputePipeline,
    kernel_buffer: wgpu::Buffer,
}

/// Device needs to have compute shader feature enabled
impl<'a> GaussianBlurGpu<'a> {
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        stdev: f32,
    ) -> GaussianBlurGpu<'a> {
        // let shader_module = ;

        // A BindGroup describes a set of resources and how they can be accessed by a shader.
        // The BindGroupLayout is kind of like an interface that defines a "type" of bind group (we could swap
        // out the bind group dynamically for one with different resources, as long as it adheres the same layout)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gaussian Blur Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gaussian Blur Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gaussian Blur"),
            layout: Some(&pipeline_layout),
            module: &device.create_shader_module(wgpu::include_wgsl!("gaussian_blur.wgsl")),
            entry_point: "entrypoint",
            compilation_options: wgpu::PipelineCompilationOptions {
                ..Default::default()
            },
        });

        let kernel = GaussianKernelGpu::new(stdev);

        let kernel_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gaussian blur kernel buffer"),
            usage: wgpu::BufferUsages::STORAGE,
            // under the hood this is doing an unsafe conversion to interpret the float data as bytes
            contents: bytemuck::cast_slice(kernel.data.as_slice()),
        });

        GaussianBlurGpu::<'a> {
            device,
            queue,
            bind_group_layout,
            compute_pipeline,
            kernel_buffer,
        }
    }

    pub fn apply(&mut self, image: &mut Image) {
        // TODO: optimise this by removing the need to copy the data to a Vec<u8> and back
        let mut unsigned_data: Vec<u8> = image.data.iter().map(|x| *x as u8).collect();
        self.shader_pass_convolve_and_tranpose_1d(&mut unsigned_data, image.width, image.height);
        self.shader_pass_convolve_and_tranpose_1d(&mut unsigned_data, image.height, image.width);
        image.data = unsigned_data.iter().map(|x| *x as i16).collect();
    }

    /// This function applies a 1D gaussian blur along each row of the image data using a compute shader.
    /// The kernel is passed as a buffer to the shader, and the image data is passed as a storage buffer.
    /// The shader reads the kernel from the buffer and the image data from the storage buffer, and writes the
    /// result to another storage buffer.
    /// We then copy the result back to the CPU and remove the padding.
    /// The shader applies the 1D convolution in the x direction on each row of the image, and write the result to
    /// the output buffer but transposed. This allows the function to be called twice to apply the 1D convolution
    /// in both the x and y directions.
    fn shader_pass_convolve_and_tranpose_1d(
        &mut self,
        data: &mut Vec<u8>,
        width: usize,
        height: usize,
    ) {
        let mut image_data_padded: Vec<u8>;
        let image_data_slice: &[u8];

        // pad the data to be a multiple of the buffer alignment
        const ALIGNMENT: usize = wgpu::COPY_BUFFER_ALIGNMENT as usize;
        let remainder = data.len() % ALIGNMENT;
        if remainder != 0 {
            image_data_padded = data.clone();
            let padding = ALIGNMENT - remainder;
            image_data_padded.extend(vec![0; padding]);
            image_data_slice = image_data_padded.as_slice();
        } else {
            // minor optimisation - we don't need to clone the data if it's already aligned
            image_data_slice = data.as_slice();
        }

        let image_buffer_size = (std::mem::size_of::<u8>() * image_data_slice.len()) as u64;

        let row_count_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gaussian blur row count buffer"),
                usage: wgpu::BufferUsages::UNIFORM,
                contents: bytemuck::cast_slice(&[width as u32]),
            });

        // CPU-accessible staging buffer for copying data into the input storage buffer
        let image_staging_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Gaussian blue staging buffer"),
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    contents: bytemuck::cast_slice(image_data_slice),
                });

        // CPU-accessible read-back buffer for copying data into from the output storage buffer
        let image_readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gaussian blur readback buffer"),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
            size: image_buffer_size,
        });

        // GPU-accessible storage buffer containing the input to process
        let image_input_storage_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gaussian blur input storage buffer buffer"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
            size: image_buffer_size,
        });

        // GPU-accessible storage buffer that the output will be written to
        let image_output_storage_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gaussian blue output storage buffer"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
            size: image_buffer_size,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gaussian blur"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.kernel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: row_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: image_input_storage_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: image_output_storage_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());

        // copy data from staging buffer to storage buffer
        encoder.copy_buffer_to_buffer(
            &image_staging_buffer,
            0,
            &image_input_storage_buffer,
            0,
            image_buffer_size,
        );

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gaussian Blur"),
                ..Default::default()
            });

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(width as u32, height as u32, 1);
        }

        // copy result to read-back buffer
        encoder.copy_buffer_to_buffer(
            &image_output_storage_buffer,
            0,
            &image_readback_buffer,
            0,
            image_readback_buffer.size(),
        );

        let command_buffer = encoder.finish();
        self.queue.submit([command_buffer]);

        // map the read-back buffer to copy the result back to CPU-accessible memory
        let buffer_slice = image_readback_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |result| match result {
            Err(e) => {
                eprintln!("failed to map buffer: {:?}", e);
            }
            Ok(_) => {}
        });

        self.device.poll(wgpu::Maintain::Wait); // TODO: poll in the background instead of blocking

        data.as_mut_slice().copy_from_slice(bytemuck::cast_slice(
            buffer_slice.get_mapped_range().split_at(width * height).0, // split at the end of the image data (remove padding)
        ));

        image_readback_buffer.unmap();
    }
}
