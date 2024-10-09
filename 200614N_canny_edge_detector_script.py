import sys
import numpy as np
from PIL import Image


def rgb_to_grayscale(rgb_image):
    r, g, b = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.uint8)


def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    # Create a 2D Gaussian kernel
    x, y = np.mgrid[-size : size + 1, -size : size + 1]
    # Calculate the normalizing constant
    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
    return kernel


def conv2d(image, kernel):
    m, n = kernel.shape
    y, x = image.shape
    y = y - m + 1
    x = x - n + 1
    new_image = np.zeros((y, x))
    # Perform 2D convolution
    for i in range(y):
        for j in range(x):
            new_image[i][j] = np.sum(image[i : i + m, j : j + n] * kernel)
    return new_image


def gaussian_blur(image, kernel_size=5, sigma=1):
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred = conv2d(image, kernel)
    return blurred


def canny_edge_detection(image):
    # Step 1: Apply Gaussian blur
    blurred = gaussian_blur(image)

    return blurred


def main():
    if len(sys.argv) != 2:
        print("Usage: python <script_name>.py <input_image>")
        sys.exit(1)

    input_name = sys.argv[1]
    output_name = input_name.rsplit(".", 1)[0] + "_edge.png"

    # Load image
    image_array = np.array(Image.open(input_name))

    # Convert image to grayscale
    if len(image_array.shape) == 3:
        gray_image = rgb_to_grayscale(image_array)
    else:
        gray_image = image_array

    # Apply Canny edge detection
    edges_array = canny_edge_detection(gray_image)

    Image.fromarray(edges_array.astype(np.uint8)).show()


if __name__ == "__main__":
    main()
