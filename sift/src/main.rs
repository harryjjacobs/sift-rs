use imageproc::drawing::{draw_hollow_circle_mut, draw_line_segment_mut};
use sift::{SiftCpu, SiftKeyPoint};

fn draw_keypoints(img: &mut image::DynamicImage, keypoints: &Vec<SiftKeyPoint>) {
    for keypoint in keypoints {
        let x = keypoint.x;
        let y = keypoint.y;
        let scale = keypoint.scale;
        let angle = keypoint.orientation;

        let x1 = x + scale * angle.cos();
        let y1 = y + scale * angle.sin();

        let color = image::Rgba([0, 255, 0, 255]);

        // if keypoint.octave == 0 {
        //     color = image::Rgba([255, 0, 0, 255]);
        // } else if keypoint.octave == 1 {
        //     color = image::Rgba([0, 255, 0, 255]);
        // } else {
        //     color = image::Rgba([0, 0, 255, 255]);
        // }

        draw_hollow_circle_mut(img, (x as i32, y as i32), scale as i32, color);
        draw_line_segment_mut(img, (x as f32, y as f32), (x1 as f32, y1 as f32), color);
    }
}

fn main() {
    println!("Hello, world!");

    let img_bytes = include_bytes!("../test_data/harbourside.png");
    let mut img = image::load_from_memory(img_bytes).unwrap();

    let mut sift = SiftCpu::new_fixed_size(img.width() as usize, img.height() as usize);
    let kps = sift.run(&img);

    println!("Found {} keypoints", kps.len());
    // for kp in &kps {
    //     println!("{}, {}", kp.layer_x, kp.layer_y);
    // }

    draw_keypoints(&mut img, &kps);
    img.save("output.png").unwrap();
}
