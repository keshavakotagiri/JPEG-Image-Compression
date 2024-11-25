import numpy as np
from scipy.fftpack import dct, idct
from skimage import io, color
import heapq
from collections import defaultdict, Counter
from huffman import huffman_decompress_binary
from binary import load_from_file
import colour_changer


def decompress_blocks(compressed_data, block_size=64, marker=32767):
    blocks = []
    current_block = []
    for num in compressed_data:
        if num == marker:  # When the EOB marker is found
            remaining_zeros = block_size - len(current_block)
            current_block.extend([0] * remaining_zeros)  # Add trailing zeros back
            blocks.append(np.array(current_block))  # Add the block to the list
            current_block = []  # Reset for the next block
        else:
            current_block.append(num)  # Add the number to the current block
    return blocks

def reverse_zigzag_order(zigzag_block):
    """
    Reverse the zigzag order and convert the 1D array back to an 8x8 block.
    """
    zigzag_indices = [
        (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
        (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
        (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
        (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
        (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
        (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
        (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
        (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
    ]
    
    # Create an empty 8x8 block
    block = np.zeros((8, 8))
    
    # Fill the block using the reverse zigzag indices
    for idx, (i, j) in enumerate(zigzag_indices):
        block[i, j] = zigzag_block[idx]
    
    return block

def apply_idct_dequantize(block, Q):
    block_dequantized = block * Q
    # Apply IDCT to each row
    block_idct_rows = idct(block_dequantized, axis=1, norm='ortho')
    # Apply IDCT to each column of the result
    block_idct_dequantized = idct(block_idct_rows, axis=0, norm='ortho')
    return block_idct_dequantized

def rejoin_blocks(blocks, num_H_blocks, num_W_blocks):
    # Create an empty array to hold the reconstructed image (size H x W)
    reconstructed_image = np.zeros((num_H_blocks * 8, num_W_blocks * 8))
    
    # Iterate through the blocks and place them in the appropriate location
    block_index = 0
    for i in range(num_H_blocks):
        for j in range(num_W_blocks):
            if block_index >= len(blocks):
                raise IndexError("Not enough blocks to fill the entire image")
            # Place each block in the correct position in the reconstructed image
            reconstructed_image[i*8:(i+1)*8, j*8:(j+1)*8] = blocks[block_index]
            block_index += 1  
    return reconstructed_image

def decompress(filename, Q, img_name, chroma_subsampling=False):
    encoded_data = load_from_file(filename)

    grey_or_col = encoded_data[0]
    encoded_data = encoded_data[1:]
    # print(type(grey_or_col))
    if(grey_or_col == '0'):
        # print("Decompressing the greyscale image ....")
        num_H_blocks = int(encoded_data[:10], 2)
        encoded_data = encoded_data[10:]
        num_W_blocks = int(encoded_data[:10], 2)
        encoded_data = encoded_data[10:]
        decompress_greyscale(encoded_data, Q, img_name, num_H_blocks, num_W_blocks, color="grey")
    else:
        # print("color_decompress", grey_or_col)
        if chroma_subsampling == True:
            num_H_blocks = int(encoded_data[:10], 2)
            encoded_data = encoded_data[10:]
            num_W_blocks = int(encoded_data[:10], 2)
            encoded_data = encoded_data[10:]
            decompress_ycbcr_color(encoded_data, Q, img_name, num_H_blocks, num_W_blocks)
        else:
            num_H_blocks = int(encoded_data[:10], 2)
            encoded_data = encoded_data[10:]
            num_W_blocks = int(encoded_data[:10], 2)
            encoded_data = encoded_data[10:]
            decompress_rgb_color(encoded_data, Q, img_name, num_H_blocks, num_W_blocks)
    return

###############################################################################################################

def decompress_greyscale(encoded_data, Q, img_name, num_H_blocks, num_W_blocks, color):
    pad_h = int(encoded_data[:3], 2)
    encoded_data = encoded_data[3:]
    pad_w = int(encoded_data[:3], 2)
    encoded_data = encoded_data[3:]

    arr = np.array(huffman_decompress_binary(encoded_data))
    blocks = decompress_blocks(arr)
    blocks = np.array(blocks)

    if len(blocks) != num_H_blocks * num_W_blocks:
        raise ValueError(f"Expected {num_H_blocks * num_W_blocks} blocks, but got {len(blocks)} blocks")
      
    reshaped_blocks = [reverse_zigzag_order(block) for block in blocks]
    idct_dequantize_blocks = [apply_idct_dequantize(block, Q) for block in reshaped_blocks]  # Apply dequantize to each block and then IDCT
    
    reconstructed_image = rejoin_blocks(idct_dequantize_blocks, num_H_blocks, num_W_blocks)

    image_to_save = np.clip(reconstructed_image, 0, 255).astype(np.uint8)  # Clip values to range 0-255 and convert to uint8

    original_height = image_to_save.shape[0] - pad_h
    original_width = image_to_save.shape[1] - pad_w
    image_to_save = image_to_save[:original_height, :original_width]
    # print(image_to_save.shape, "->", color)
    # Save the image using skimage.io.imsave
    if color == "grey":
        img_name += "_compressed.jpg"
    else:
        img_name += "_" + color + "_compressed.jpg" # remove the .bin at the end and add .jpg
    io.imsave(img_name, image_to_save)

    # print("compressed image saved as:", img_name)
    return image_to_save

def decompress_rgb_color(encoded_data, Q, img_name, num_H_blocks, num_W_blocks):
    pad_h = int(encoded_data[:3], 2)
    encoded_data = encoded_data[3:]
    pad_w = int(encoded_data[:3], 2)
    encoded_data = encoded_data[3:]

    arr = np.array(huffman_decompress_binary(encoded_data))
    # print(arr.shape, "->", color)
    blocks = decompress_blocks(arr)
    blocks = np.array(blocks)
    # print("length of the decompressed blocks is:", len(blocks))
    # Split the blocks into 3 sub-parts for R, G and B
    total_blocks = len(blocks)
    sub_part_size = total_blocks // 3
    if total_blocks % 3 != 0:
        raise ValueError("Cannot split blocks into 3 equal parts")

    sub_parts = {
        'red': blocks[:sub_part_size],
        'green': blocks[sub_part_size:2*sub_part_size],
        'blue': blocks[2*sub_part_size:]
    }

    red_decompressed = []
    green_decompressed = []
    blue_decompressed = []

    for color, my_blocks in sub_parts.items():
        # print(f"colour {color} shape: {np.array(my_blocks).shape}")
        reshaped_blocks = [reverse_zigzag_order(block) for block in my_blocks]
        idct_dequantize_blocks = [apply_idct_dequantize(block, Q) for block in reshaped_blocks]
        reconstructed_image = rejoin_blocks(idct_dequantize_blocks, num_H_blocks, num_W_blocks)
        image_to_save = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

        original_height = image_to_save.shape[0] - pad_h
        original_width = image_to_save.shape[1] - pad_w
        image_to_save = image_to_save[:original_height, :original_width]

        if color == "red":
            red_decompressed = image_to_save
        elif color == "green":
            green_decompressed = image_to_save
        else:
            blue_decompressed = image_to_save

        # print(image_to_save.shape, "->", color)
        file_name = img_name
        file_name += "_" + color + "_compressed.jpg" # remove the .bin at the end and add .jpg
        io.imsave(file_name, image_to_save)

    # Stack the 3 decompressed channels to form the final color image
    final_image = np.stack((red_decompressed, green_decompressed, blue_decompressed), axis=-1)

    # Save the final color image using skimage.io.imsave
    compressed_img_name = img_name + "_rgb_compressed.jpg"

    io.imsave(compressed_img_name, final_image)
    return final_image


def decompress_ycbcr_color(encoded_data, Q, img_name, num_H_blocks, num_W_blocks):
    pad_h = int(encoded_data[:3], 2)
    encoded_data = encoded_data[3:]
    pad_w = int(encoded_data[:3], 2)
    encoded_data = encoded_data[3:]

    arr = np.array(huffman_decompress_binary(encoded_data))
    blocks = decompress_blocks(arr)
    blocks = np.array(blocks)
    # print("length of the decompressed blocks is:", len(blocks))
    # Split the blocks into 3 sub-parts for R, G and B
    total_blocks = len(blocks)
    # print(total_blocks)
    # print(num_H_blocks*num_W_blocks)
    y_blocks = total_blocks * 4 // 6  # Y gets 4 parts out of 6
    cb_cr_blocks = total_blocks // 6  # Cb and Cr each get 1 part out of 6
    # print(y_blocks, cb_cr_blocks, cb_cr_blocks)
    sub_parts = {
        'Y': blocks[:y_blocks],
        'Cb': blocks[y_blocks:y_blocks + cb_cr_blocks],
        'Cr': blocks[y_blocks + cb_cr_blocks:]
    }

    y_decompressed = []
    cb_decompressed = []
    cr_decompressed = []
    # print("number of blocks H and W are: ", num_H_blocks, num_W_blocks)
    for color, my_blocks in sub_parts.items():
        # print(f"colour {color} shape: {np.array(my_blocks).shape}")
        reshaped_blocks = [reverse_zigzag_order(block) for block in my_blocks]
        idct_dequantize_blocks = [apply_idct_dequantize(block, Q) for block in reshaped_blocks]
        # print("length of the idct_dequantize_blocks is:", len(idct_dequantize_blocks))

        if color == "Y":
            reconstructed_image = rejoin_blocks(idct_dequantize_blocks, num_H_blocks, num_W_blocks)
            # print(reconstructed_image.shape)
            image_to_save = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

            original_height = image_to_save.shape[0] - pad_h
            original_width = image_to_save.shape[1] - pad_w
            image_to_save = image_to_save[:original_height, :original_width]
            y_decompressed = image_to_save

        elif color == "Cb":
            reconstructed_image = rejoin_blocks(idct_dequantize_blocks, num_H_blocks//2, num_W_blocks//2)
            # print(reconstructed_image.shape)
            image_to_save = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

            original_height = image_to_save.shape[0] - pad_h
            original_width = image_to_save.shape[1] - pad_w
            image_to_save = image_to_save[:original_height, :original_width]
            cb_decompressed = image_to_save
        else:
            reconstructed_image = rejoin_blocks(idct_dequantize_blocks, num_H_blocks//2, num_W_blocks//2)
            # print(reconstructed_image.shape)
            image_to_save = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

            original_height = image_to_save.shape[0] - pad_h
            original_width = image_to_save.shape[1] - pad_w
            image_to_save = image_to_save[:original_height, :original_width]
            cr_decompressed = image_to_save

        # print(image_to_save.shape, "->", color)
        file_name = img_name
        file_name += "_" + color + "_compressed.jpg" # remove the .bin at the end and add .jpg
        io.imsave(file_name, image_to_save)

    # Stack the 3 decompressed channels to form the final color image
    Y_shape = y_decompressed.shape 
    Cb_upsampled = colour_changer.upsample_channel(cb_decompressed, (Y_shape[0], Y_shape[1])).astype(np.uint8)
    Cr_upsampled = colour_changer.upsample_channel(cr_decompressed, (Y_shape[0], Y_shape[1])).astype(np.uint8)
    final_image = np.stack((y_decompressed, Cb_upsampled, Cr_upsampled), axis=-1)
    final_image = colour_changer.ycbcr_to_rgb(final_image)

    # Save the final color image using skimage.io.imsave
    compressed_img_name = img_name + "_ycbcr_compressed.jpg"
    io.imsave(compressed_img_name, final_image)
    return final_image