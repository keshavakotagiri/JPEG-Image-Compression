import os
import user
import numpy as np
import matplotlib.pyplot as plt

option = input("Enter 1 for greyscale images and 2 for color images: ")
if option == "1":
    image_dir = './gray_images_dataset'
elif option == "2":
    image_dir = './color_images_dataset'
# Directory containing images


# List to store RMSE and BPP values
rmse_values = []
bpp_values = []

Q_values = [10, 30, 50, 70, 90]  # Quality factors to test

# Iterate over all files in the directory
for img_name in os.listdir(image_dir):
    if img_name.endswith('.jpg') or img_name.endswith('.png'):  # Add other image formats if needed
        print(img_name)
        img_path = os.path.join(image_dir, img_name)
        get_rmse_for_image = []
        get_bpp_for_image = []
        for Q in Q_values:  # Add other sampling values if needed
            sampling = "no"
            if(option == "1"):
                sampling = "no"
            elif(option == "2"):
                sampling = "yes"
            # print(sampling)
            rmse_val, bpp_val = user.menu(img_path, Q, sampling)  # Assuming user.py has a function menu that returns RMSE and BPP
            get_rmse_for_image.append(rmse_val)
            get_bpp_for_image.append(bpp_val)

        rmse_values.append(get_rmse_for_image)
        bpp_values.append(get_bpp_for_image)

# Plotting the RMSE vs BPP values
plt.figure(figsize=(10, 6))
colors = [
    'b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'lime', 'indigo', 'violet', 'gold', 'silver'
]
color_index = 0

for rmse_list, bpp_list in zip(rmse_values, bpp_values):
    plt.plot(bpp_list, rmse_list, 'o-', color=colors[color_index % len(colors)], label=f"Image {color_index + 1}")
    color_index += 1

plt.xlabel("Bits Per Pixel (BPP)")
plt.ylabel("Root Mean Square Error (RMSE)")
plt.title("RMSE vs BPP for JPEG Compression For Greyscale Images")
plt.legend()
plt.grid(True)
plt.savefig("RMSE_vs_BPP_grayscale.png")
plt.show()



