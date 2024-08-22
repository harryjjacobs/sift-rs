use imageproc::drawing::draw_line_segment_mut;
use sift::{SiftCpu, SiftKeyPoint};

fn draw_keypoints(img: &mut image::DynamicImage, keypoints: &Vec<SiftKeyPoint>) {
    for keypoint in keypoints {
        let x = keypoint.x as f32;
        let y = keypoint.y as f32;
        let scale = 10.0;
        let angle: f32 = 0.0;

        let x1 = x + scale * angle.cos();
        let y1 = y + scale * angle.sin();

        draw_line_segment_mut(
            img,
            (x as f32, y as f32),
            (x1, y1),
            image::Rgba([0, 255, 0, 255]),
        );
    }
}

fn main() {
    println!("Hello, world!");

    let img_bytes = include_bytes!("../test_data/harbourside.png");
    let mut img = image::load_from_memory(img_bytes).unwrap();

    let mut sift = SiftCpu::new_fixed_size(img.width() as usize, img.height() as usize);
    let kps = sift.run(&img);

    println!("Found {} keypoints", kps.len());
    for kp in &kps {
        println!("{}, {}", kp.x, kp.y);
    }

    draw_keypoints(&mut img, &kps);
    img.save("output.png").unwrap();
}
