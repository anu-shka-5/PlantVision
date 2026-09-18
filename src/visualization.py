import cv2
import numpy as np
import matplotlib.pyplot as plt


def save_image(path, image):
    """
    Save an image to disk.
    """
    success = cv2.imwrite(path, image)

    if not success:
        raise IOError(
            f"Could not save image: {path}"
        )


def normalize_image(image):
    """
    Convert an image to displayable 8-bit format.
    """
    image = np.asarray(image)

    if image.dtype == np.uint8:
        return image

    min_value = image.min()
    max_value = image.max()

    if max_value == min_value:
        return np.zeros_like(
            image,
            dtype=np.uint8
        )

    normalized = (
        (image - min_value)
        / (max_value - min_value)
        * 255
    )

    return normalized.astype(np.uint8)


def save_hog_visualization(
    hog_image,
    path
):
    """
    Save HOG visualization.
    """
    hog_image = normalize_image(
        hog_image
    )

    save_image(
        path,
        hog_image
    )


def save_feature_visualization(
    images,
    titles,
    path
):
    """
    Save a combined visualization.
    """
    columns = 2
    rows = (len(images) + 1) // 2

    plt.figure(
        figsize=(12, 5 * rows)
    )

    for i, (image, title) in enumerate(
        zip(images, titles)
    ):

        plt.subplot(
            rows,
            columns,
            i + 1
        )

        if len(image.shape) == 2:
            plt.imshow(
                image,
                cmap="gray"
            )
        else:
            plt.imshow(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )
            )

        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=150
    )

    plt.close()