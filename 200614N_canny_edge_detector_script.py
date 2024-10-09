import sys
import numpy as np
from PIL import Image


def rgb_to_grayscale(rgb_image):
    r, g, b = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.uint8)


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


if __name__ == "__main__":
    main()
