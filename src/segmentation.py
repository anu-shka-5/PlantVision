import cv2
import numpy as np


def otsu_segmentation(gray):
    """
    Segment the image using Otsu thresholding.
    """
    threshold, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return threshold, binary


def grabcut_segmentation(image):
    """
    Foreground extraction using GrabCut.
    """
    mask = np.zeros(
        image.shape[:2],
        np.uint8
    )

    height, width = image.shape[:2]

    margin_x = max(5, width // 20)
    margin_y = max(5, height // 20)

    rect = (
        margin_x,
        margin_y,
        width - 2 * margin_x,
        height - 2 * margin_y
    )

    background_model = np.zeros(
        (1, 65),
        np.float64
    )

    foreground_model = np.zeros(
        (1, 65),
        np.float64
    )

    cv2.grabCut(
        image,
        mask,
        rect,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    foreground_mask = np.where(
        (mask == 2) | (mask == 0),
        0,
        255
    ).astype("uint8")

    return foreground_mask


def apply_mask(image, mask):
    """
    Apply a binary foreground mask.
    """
    return cv2.bitwise_and(
        image,
        image,
        mask=mask
    )


def region_growing(gray, seed=None, tolerance=15):
    """
    Simple region growing demonstration.

    A seed is selected automatically if none is supplied.
    """
    height, width = gray.shape

    if seed is None:
        seed = (
            height // 2,
            width // 2
        )

    seed_y, seed_x = seed

    visited = np.zeros_like(
        gray,
        dtype=np.uint8
    )

    region = np.zeros_like(
        gray,
        dtype=np.uint8
    )

    seed_value = int(
        gray[seed_y, seed_x]
    )

    queue = [(seed_y, seed_x)]

    while queue:

        y, x = queue.pop(0)

        if y < 0 or y >= height:
            continue

        if x < 0 or x >= width:
            continue

        if visited[y, x]:
            continue

        visited[y, x] = 1

        if abs(
            int(gray[y, x]) - seed_value
        ) <= tolerance:

            region[y, x] = 255

            queue.extend([
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1)
            ])

    return region


def foreground_percentage(mask):
    """
    Calculate percentage of foreground pixels.
    """
    return round(
        100 * np.count_nonzero(mask) / mask.size,
        2
    )