import cv2


def load_image(path):
    """
    Load an image from the supplied path.
    """
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {path}"
        )

    return image


def resize_image(image, size=(512, 512)):
    """
    Resize image to a fixed size.
    """
    return cv2.resize(image, size)


def convert_to_gray(image):
    """
    Convert BGR image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(gray, kernel_size=5):
    """
    Reduce noise before edge detection.
    """
    return cv2.GaussianBlur(
        gray,
        (kernel_size, kernel_size),
        0
    )