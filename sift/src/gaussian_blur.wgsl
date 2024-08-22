@group(0)
@binding(0)
var<storage> kernel: array<f32>;

@group(0)
@binding(1)
// The number of u8 values per row packed into the input buffer (there might be padding at the end of input)
var<uniform> rowWidth: u32;

@group(0)
@binding(2)
var<storage, read> input: array<u32>;

@group(0)
@binding(3)
var<storage, read_write> output: array<atomic<u32>>;

const MASK: u32 = 0xFF;

@compute
@workgroup_size(1)
fn entrypoint(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
    let numRows = (arrayLength(&input) << 2u) / rowWidth;
    let kernelSize = arrayLength(&kernel);
    let kernelRadius = kernelSize >> 1u;
    let xMax = i32(rowWidth) - 1;

     // Calculate the starting offset for the current row
    let rowOffset = global_invocation_id.y * rowWidth;
    let x = global_invocation_id.x;
    
    // Loop through each byte in the row
    var sum = 0.0;
    for (var i = 0u; i < kernelSize; i++) {
            // Calculate the x coordinate in the input buffer
        var xx = i32(x) - i32(kernelRadius) + i32(i);

            // Reflect the x coordinate if it is out of bounds
        if xx < 0 {
            xx = -xx;
        } else if xx > xMax {
            xx = xMax - (xx - xMax);
        }

            // Calculate the index in the input array
        let byteIndex = rowOffset + u32(xx);
        let wordIndex = byteIndex >> 2u;
        let byteOffsetInWord = byteIndex % 4u;

            // Calculate the shift for the byte within the u32 word
        let shift = byteOffsetInWord << 3u;

            // Extract the byte value from the input
        let value = (input[wordIndex] >> shift) & MASK;
        sum += f32(value) * kernel[i];
    }

    sum = clamp(sum, 0.0, 255.0);

        // Transpose the image (swap rows and columns) in this 1d pass, so that the next pass can be done in the same way
    let byteIndex = x * numRows + (global_invocation_id.y);
    let wordIndex = byteIndex >> 2u;
    let byteOffsetInWord = byteIndex % 4u;

    // Calculate the shift for the byte within the u32 word
    let shift = byteOffsetInWord << 3u;

        // Clear the byte in the output
    atomicAnd(&output[wordIndex], ~(MASK << shift));

        // Set the byte in the output
    atomicOr(&output[wordIndex], (u32(sum) << shift));
}
