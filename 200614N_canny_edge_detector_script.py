import sys
import numpy as np
from PIL import Image


def main():
    if len(sys.argv) != 2:
        print("Usage: python <script_name>.py <input_image>")
        sys.exit(1)

    input_name = sys.argv[1]
    output_name = input_name.rsplit(".", 1)[0] + "_edge.png"

    # Load image
    image_array = np.array(Image.open(input_name))


if __name__ == "__main__":
    main()
