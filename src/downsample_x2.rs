use crate::sift_image::Image;

impl Image {
    pub fn downsample_x2(&mut self) {
        // let original = self.clone();
        let downsampled_width = self.width / 2;
        let downsampled_height = self.height / 2;
        for y in 0..downsampled_height {
            for x in 0..downsampled_width {
                self.data[downsampled_width * y + x] = self.data[self.width * y * 2 + x * 2];
            }
        }
        self.width = downsampled_width;
        self.height = downsampled_height;
        self.data
            .resize(downsampled_width * downsampled_height, Default::default());
    }
}
