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


def sobel_filters(img):
    # Sobel kernels
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    # Convolve kernels with image
    Ix = conv2d(img, Kx)
    Iy = conv2d(img, Ky)

    # Calculate gradient magnitude and direction
    G = np.hypot(Ix, Iy)
    # Normalize gradient values to 0-255
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    return G, theta


def non_max_suppression(img, theta):
    # Get image dimensions and angles in degrees
    M, N = img.shape
    output_img = np.zeros((M, N), dtype=np.int32)
    angle = theta * 180.0 / np.pi
    angle[angle < 0] += 180

    # Perform non-maximum suppression
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]  # Right pixel
                    r = img[i, j - 1]  # Left pixel
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]  # Bottom-left pixel
                    r = img[i - 1, j + 1]  # Top-right pixel
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]  # Bottom pixel
                    r = img[i - 1, j]  # Top pixel
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]  # Bottom-right pixel
                    r = img[i + 1, j + 1]  # Top-left pixel

                # Suppress non-maximum pixels
                if (img[i, j] >= q) and (img[i, j] >= r):
                    output_img[i, j] = img[i, j]
                else:
                    output_img[i, j] = 0

            except IndexError as e:
                pass

    return output_img


def canny_edge_detection(image):
    # Step 1: Apply Gaussian blur
    blurred = gaussian_blur(image)

    # Step 2: Calculate gradients
    gradients, theta = sobel_filters(blurred)

    # Step 3: Non-maximum suppression
    suppressed = non_max_suppression(gradients, theta)

    return suppressed


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
