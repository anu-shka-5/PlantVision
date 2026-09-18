import cv2
import numpy as np


def canny_edges(gray, low_threshold=80, high_threshold=160):
    """
    Detect edges using the Canny edge detector.
    """
    return cv2.Canny(
        gray,
        low_threshold,
        high_threshold
    )


def log_edges(gray, sigma=1.0):
    """
    Laplacian of Gaussian edge detection.
    """
    blurred = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigma
    )

    laplacian = cv2.Laplacian(
        blurred,
        cv2.CV_64F
    )

    laplacian = np.absolute(laplacian)

    if laplacian.max() > 0:
        laplacian = (
            laplacian / laplacian.max() * 255
        )

    return laplacian.astype(np.uint8)


def difference_of_gaussian(gray, sigma1=1.0, sigma2=2.0):
    """
    Difference of Gaussian edge/detail response.
    """
    blur1 = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigma1
    )

    blur2 = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigma2
    )

    dog = cv2.absdiff(blur1, blur2)

    return dog


def hough_lines(edges):
    """
    Detect line segments using the probabilistic Hough transform.
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=30,
        maxLineGap=10
    )

    return lines


def draw_hough_lines(image, lines):
    output = image.copy()

    if lines is not None:
        for line in lines:
            line = line.reshape(-1)

            if len(line) >= 4:
                x1, y1, x2, y2 = line[:4]

                cv2.line(
                    output,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

    return output


def harris_corners(gray):
    """
    Detect corners using the Harris corner detector.
    """
    gray_float = np.float32(gray)

    response = cv2.cornerHarris(
        gray_float,
        blockSize=2,
        ksize=3,
        k=0.04
    )

    response = cv2.dilate(response, None)

    return response


def draw_harris_corners(image, response):
    """
    Mark strong Harris corner responses.
    """
    output = image.copy()

    threshold = 0.01 * response.max()

    output[response > threshold] = [0, 0, 255]

    return output