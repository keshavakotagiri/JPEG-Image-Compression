from compress import compress
from decompress import decompress
import colour_changer
import numpy as np
from skimage import io, color
import os

def generate_quant_matrix(q):
    Q = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    return np.round(Q * (50/q))

def calculate_bpp(compressed_file_path, image_shape):
    """Calculate BPP from the compressed file size and image dimensions."""
    file_size_bits = 8 * os.path.getsize(compressed_file_path)  # File size in bits
    total_pixels = image_shape[0] * image_shape[1]
    return file_size_bits / total_pixels

def calculate_rmse_color(image1, image2):
    # Ensure the images are numpy arrays
    image1 = np.array(image1)
    image2 = np.array(image2)

    # Check if the images have the same shape
    
    if image1.shape != image2.shape:
        # print(image1.shape)
        # print(image2.shape)
        raise ValueError("Images must have the same dimensions")

    # Calculate the squared differences between the images
    squared_diff = (image1 - image2) ** 2

    # Calculate the MSE for each channel
    mse = np.mean(squared_diff, axis=(0, 1))

    # Calculate the RMSE for each channel
    rmse = np.sqrt(mse)

    # Normalize the RMSE by the mean of the original image
    rmse /= np.mean(image1, axis=(0, 1))

    # Return the average RMSE across all channels
    return np.mean(rmse)

def calculate_rmse(image1, image2):
    # Ensure the images are numpy arrays
    image1 = np.array(image1)
    image2 = np.array(image2)

    # Check if the images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")

    # Calculate the squared differences between the images
    squared_diff = (image1 - image2) ** 2

    # Calculate the MSE
    mse = np.mean(squared_diff)
    rmse = np.sqrt(mse)
    rmse /= np.mean(image1)
    return rmse

def menu(image_path, quality_factor, my_sampling):
    rmse_val = 0
    bpp_val = 0
    img_name = image_path.split(".")[0]
    quant_matrix = generate_quant_matrix(quality_factor)

    img = io.imread(image_path)
    shape = img.shape
    # print("shape of given image is:", shape)

    if(len(shape) == 3):
        if my_sampling == "yes":
            chroma_subsampling = True
            compress(image_path, quant_matrix, chroma_subsampling=True)
            ycbcr_bin = f"{img_name}_ycbcr.bin"
            decompress(ycbcr_bin, quant_matrix, img_name, chroma_subsampling=True)
            final_img = io.imread(f"{img_name}_ycbcr_compressed.jpg")
            rmse_val = calculate_rmse_color(img, final_img)
            bpp_val = calculate_bpp(f"{img_name}_ycbcr_compressed.jpg", img.shape)

        else:
            compress(image_path, quant_matrix, chroma_subsampling=False)
            rgb_bin = f"{img_name}_rgb.bin"           
            decompress(rgb_bin, quant_matrix, img_name, chroma_subsampling=False)

            final_img = io.imread(f"{img_name}_rgb_compressed.jpg")
            rmse_val = calculate_rmse_color(img, final_img)
            bpp_val = calculate_bpp(f"{img_name}_rgb_compressed.jpg", img.shape)

    elif(len(shape) == 2):
        compress(image_path, quant_matrix, chroma_subsampling=False)
        binary_path = f"{img_name}_grey.bin"
        decompress(binary_path, quant_matrix, img_name, chroma_subsampling=False)
        
        image1 = io.imread(image_path)
        image2 = io.imread(f"{img_name}_compressed.jpg")
        rmse_val = calculate_rmse(image1, image2)
        bpp_val = calculate_bpp(f"{img_name}_compressed.jpg", img.shape)
        

    print(f"RMSE value for image {img_name} with compressed image is: {rmse_val}")
    print(f"BPP value is {bpp_val}")
    return rmse_val, bpp_val

if __name__ == "__main__":
    # options = input("do you want to run for a single image or generate RMSE vs BPP plot? (single/plot): ")
    image_path = input("Enter image name: ")  # Set the image path
    my_img = io.imread(image_path)
    quality_factor = int(input("Enter compression value (integer in range [1-100]) : "))  
    if len(my_img.shape) == 3:
        my_sampling = input("Do you want to compress the image using chroma subsampling? (yes/no): ")
    else:
        my_sampling = "no"

    bpp_val, rmse_val = menu(image_path, quality_factor, my_sampling)
    # print(bpp_val, rmse_val)
    

    



