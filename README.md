# JPEG Image Compression
- There is a report pdf as well other than this README, in which the project is explained in detail along with a plot.

## Overview
This project implements JPEG image compression for both color and grayscale images. The compression algorithm uses the Discrete Cosine Transform (DCT) to retain essential image details while discarding less noticeable information. The compressed data is stored in a binary format and can be decompressed to reconstruct the image, albeit with some loss of quality due to the lossy nature of JPEG compression.

The project evaluates the compression efficiency using metrics like Root Mean Squared Error (RMSE) and Bits Per Pixel (BPP) by analyzing 20 sample images.

---

## Features
- Compress and decompress color and grayscale images.
- Support for RGB and YCbCr color spaces.
- Implementation of key JPEG steps:
  - Block division
  - DCT transformation
  - Quantization
  - Huffman encoding
  - Zig-zag ordering
  - Inverse DCT for reconstruction
- Generates RMSE vs. BPP plots for image quality analysis.

---

## File Structure
Submission/ ├── color_images_dataset/ ├── gray_images_dataset/ ├── binary.py ├── compress.py ├── decompress.py ├── huffman.py ├── user.py ├── generate_graph.py ├── run_compressor.sh ├── run_graph.sh ├── kodak24.jpg ├── kodim24.png ├── report.pdf


---

## Code Components

### 1. compress.py
- Compresses an image into a binary `.bin` file.
- Steps:
  - Converts the image into 8x8 blocks.
  - Applies DCT to each block.
  - Quantizes the DCT coefficients.
  - Uses zig-zag ordering and Huffman encoding for efficient storage.

### 2. huffman.py
- Implements Huffman encoding and decoding:
  - `huffman_compress()`: Encodes the data into a bit string.
  - `huffman_decompress()`: Decodes the bit string back to the original data.

### 3. binary.py
- Handles saving and loading `.bin` files:
  - Adds padding for bit alignment.
  - Removes padding during decompression.

### 4. decompress.py
- Reverses the compression steps:
  - Decodes the binary file.
  - Reconstructs image blocks using IDCT.
  - Reassembles the image.

### 5. user.py
- Provides an interface for testing the JPEG algorithm.
- Generates RMSE vs. BPP plots for quality evaluation.

---

## Color Image Handling
- Supports two methods:
  1. **RGB Compression**: Compresses each channel (R, G, B) separately.
  2. **YCbCr Compression**:
     - Converts RGB to YCbCr.
     - Downsamples Cb and Cr channels.
     - Compresses Y, Cb, and Cr channels separately.
     - Resamples Cb and Cr during decompression.

---

## Results
- Tested on 20 grayscale and 20 color images.
- Quality factors: `{10, 30, 50, 70, 90}`.
- RMSE vs. BPP plots demonstrate the trade-off between compression ratio and quality.

---

## Getting Started

### Dependencies
- Python 3.x
- Libraries: `numpy`, `matplotlib`, `Pillow`

### Usage

1. **Compress an Image**:
   ```bash
   python compress.py <input_image> <quantization_matrix>
2. **Decompress an Image:**
   ```bash
   python decompress.py <compressed_file>
3. **Generate RMSE vs. BPP Plot:**
   ```bash
   python generate_graph.py
5. **Run Compression Script:**
   ```bash
   sh run_compressor.sh
7. **Run Graph Plot Script:**
   ```bash
   sh run_graph.sh
