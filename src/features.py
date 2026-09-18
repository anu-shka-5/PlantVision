import cv2
import numpy as np
from skimage.feature import hog


def extract_sift_features(gray):
    """
    Extract local features using SIFT.
    """
    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(
        gray,
        None
    )

    return keypoints, descriptors


def draw_sift_features(image, keypoints):
    """
    Draw SIFT keypoints on the image.
    """
    return cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )


def extract_hog(gray):
    """
    Extract Histogram of Oriented Gradients.
    """
    features, visualization = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        block_norm="L2-Hys"
    )

    return features, visualization


def gabor_features(gray):
    """
    Generate Gabor responses at several orientations.
    """
    gray_float = gray.astype(np.float32) / 255.0

    responses = []

    for theta in [
        0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4
    ]:

        kernel = cv2.getGaborKernel(
            (21, 21),
            sigma=4.0,
            theta=theta,
            lambd=10.0,
            gamma=0.5,
            psi=0,
            ktype=cv2.CV_32F
        )

        response = cv2.filter2D(
            gray_float,
            cv2.CV_32F,
            kernel
        )

        responses.append(response)

    return responses


def haar_dwt(gray):
    """
    Single-level Haar wavelet decomposition.
    """
    image = gray.astype(np.float32)

    if image.shape[0] % 2 != 0:
        image = image[:-1, :]

    if image.shape[1] % 2 != 0:
        image = image[:, :-1]

    top = image[0::2, :]
    bottom = image[1::2, :]

    approx = (top + bottom) / 2
    detail_vertical = (top - bottom) / 2

    left = approx[:, 0::2]
    right = approx[:, 1::2]

    approx_final = (left + right) / 2
    detail_horizontal = (left - right) / 2

    left_detail = detail_vertical[:, 0::2]
    right_detail = detail_vertical[:, 1::2]

    detail_diagonal = (
        left_detail - right_detail
    ) / 2

    detail_vertical = (
        left_detail + right_detail
    ) / 2

    return (
        approx_final,
        detail_horizontal,
        detail_vertical,
        detail_diagonal
    )


def calculate_feature_statistics(
    gray,
    edges,
    keypoints,
    hog_features,
    gabor_responses
):
    """
    Generate numerical feature statistics.
    """

    statistics = {}

    statistics["image_height"] = gray.shape[0]
    statistics["image_width"] = gray.shape[1]

    statistics["edge_pixels"] = int(
        np.count_nonzero(edges)
    )

    statistics["edge_percentage"] = round(
        100 * np.count_nonzero(edges) / edges.size,
        2
    )

    statistics["sift_keypoints"] = len(keypoints)

    statistics["hog_feature_length"] = len(
        hog_features
    )

    statistics["gabor_orientations"] = len(
        gabor_responses
    )

    statistics["gabor_mean_response"] = round(
        float(
            np.mean(
                [
                    np.mean(np.abs(r))
                    for r in gabor_responses
                ]
            )
        ),
        4
    )

    return statistics