import cv2
import numpy as np

def resize_image(image, target_shape, mode='resize', padding_color=(0, 0, 0)):
    """
    Resize an image into a given shape in two ways:
    1) Directly interpolate and resize.
    2) Scale the image size but keep the original ratio, and pad the rest regions.

    Parameters:
    - image: np.array, the input image.
    - target_shape: tuple, the target shape (height, width).
    - mode: str, 'interpolate' or 'scale_and_pad'.
    - padding_color: tuple, the color for padding regions (B, G, R).

    Returns:
    - resized_image: np.array, the resized image.
    """
    target_height, target_width = target_shape

    if mode == 'resize':
        # Directly interpolate and resize
        resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    elif mode == 'pad':
        # Scale the image size but keep the original ratio, and pad the rest regions
        original_height, original_width = image.shape[:2]
        aspect_ratio = original_width / original_height

        if target_width / target_height > aspect_ratio:
            # Fit to height
            new_height = target_height
            new_width = int(aspect_ratio * target_height)
        else:
            # Fit to width
            new_width = target_width
            new_height = int(target_width / aspect_ratio)

        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Create a new image with the target shape and fill it with padding color
        padded_image = np.full((target_height, target_width, 3), padding_color, dtype=np.uint8)

        # Calculate padding offsets
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2

        # Place the resized image in the center of the padded image
        padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
        resized_image = padded_image
    else:
        raise ValueError("Mode should be either 'interpolate' or 'scale_and_pad'")

    return resized_image