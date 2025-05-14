from PIL import Image
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import os

def detect_staff_lines(input_path, output_path, 
                       gray_threshold=250, 
                       horizontal_kernel_width=100, 
                       line_response_threshold=10, 
                       min_black_pixel_ratio=0.5):
    """
    Detects staff lines in a grayscale music sheet image and visualizes them by drawing red lines.

    Parameters:
        input_path (str): Path to the input image (music sheet).
        output_path (str): Path to save the output image with detected staff lines highlighted.
        gray_threshold (int): Pixel intensity threshold to binarize the image. Pixels darker than this are considered "black".
        horizontal_kernel_width (int): Width of the horizontal kernel used to detect horizontal line patterns.
        line_response_threshold (int): Convolution response threshold to determine if a row contains a potential staff line.
        min_black_pixel_ratio (float): Minimum ratio of black pixels required in a row to be considered valid.
    """

    # 1. Load image and convert to grayscale
    img = Image.open(input_path).convert('L')
    img_np = np.array(img)
    binary = (img_np < gray_threshold).astype(np.uint8) #Pixels with values below gray_threshold are considered black (1), others white (0)

    # 2. Create a 1D horizontal kernel to scan across image rows
    horizontal_kernel = np.ones((1, horizontal_kernel_width), dtype=np.uint8)
    response = convolve(binary, horizontal_kernel)

    # 3. Mark rows as line candidates if their response exceeds the set threshold
    line_candidates = (response > line_response_threshold).astype(np.uint8)

    # 4. Filter line candidates based on the number of black pixels in each row
    image_width = img_np.shape[1]
    min_black_pixels_per_row = int(min_black_pixel_ratio * image_width)

    valid_line_y_coords = [
        y for y in range(binary.shape[0])
        if np.max(line_candidates[y]) > 0 and np.sum(binary[y]) >= min_black_pixels_per_row
    ]

    # 5. Visualize the result
    output_rgb = np.stack([img_np]*3, axis=-1)
    for y in valid_line_y_coords:
        output_rgb[y, :, 0] = 255  # Draw detected lines in red (R=255, G=0, B=0)
        output_rgb[y, :, 1] = 0
        output_rgb[y, :, 2] = 0

    # 6. Save output image with highlighted staff lines
    result_img = Image.fromarray(output_rgb)
    result_img.save(output_path)
    print(f"[✓] Music Sheet with Staff Lines are saved to {output_path}")



if __name__ == "__main__":

    #1. Define Input and output directories
    input_dir = "./sheet_samples"
    output_dir = "./staffline_detection"
    os.makedirs(output_dir, exist_ok=True)

    #2. Specify valid files
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}

    #3. Process all images in the directory
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in valid_exts):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            detect_staff_lines(input_path, output_path)