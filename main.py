import argparse
import os

import cv2
import numpy as np

from src.preprocessing import (
    load_image,
    resize_image,
    convert_to_gray,
    gaussian_blur
)

from src.edges import (
    canny_edges,
    log_edges,
    difference_of_gaussian,
    hough_lines,
    draw_hough_lines,
    harris_corners,
    draw_harris_corners
)

from src.features import (
    extract_sift_features,
    draw_sift_features,
    extract_hog,
    gabor_features,
    haar_dwt,
    calculate_feature_statistics
)

from src.segmentation import (
    otsu_segmentation,
    grabcut_segmentation,
    apply_mask,
    region_growing,
    foreground_percentage
)

from src.visualization import (
    save_image,
    save_hog_visualization
)


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "PlantVision - Classical Computer "
            "Vision Based Plant Image Analysis"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input image"
    )

    parser.add_argument(
        "--method",
        default="all",
        choices=[
            "all",
            "edges",
            "features",
            "segmentation"
        ],
        help="Processing method"
    )

    parser.add_argument(
        "--output",
        default="results",
        help="Output directory"
    )

    return parser.parse_args()


def save_summary(
    statistics,
    segmentation_percentage,
    path
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "PLANTVISION ANALYSIS SUMMARY\n"
        )

        file.write(
            "============================\n\n"
        )

        for key, value in statistics.items():

            readable = key.replace(
                "_",
                " "
            ).title()

            file.write(
                f"{readable}: {value}\n"
            )

        file.write(
            f"\nGrabCut Foreground Percentage: "
            f"{segmentation_percentage}%\n"
        )


def main():

    args = parse_arguments()

    os.makedirs(
        args.output,
        exist_ok=True
    )

    print()
    print("=" * 55)
    print("                  PLANTVISION")
    print("     Classical Computer Vision Analysis")
    print("=" * 55)

    print(
        f"\nInput image: {args.input}"
    )

    image = load_image(
        args.input
    )

    image = resize_image(
        image,
        (512, 512)
    )

    gray = convert_to_gray(
        image
    )

    blurred = gaussian_blur(
        gray
    )

    save_image(
        os.path.join(
            args.output,
            "01_original.jpg"
        ),
        image
    )

    save_image(
        os.path.join(
            args.output,
            "02_grayscale.jpg"
        ),
        gray
    )

    print(
        f"Image size: "
        f"{gray.shape[1]} x {gray.shape[0]}"
    )

    # ------------------------------------------------
    # EDGE PROCESSING
    # ------------------------------------------------

    edges = canny_edges(
        blurred
    )

    if args.method in [
        "all",
        "edges"
    ]:

        print(
            "\n[1] Running edge detection..."
        )

        log_result = log_edges(
            gray
        )

        dog_result = difference_of_gaussian(
            gray
        )

        lines = hough_lines(
            edges
        )

        hough_result = draw_hough_lines(
            image,
            lines
        )

        corners = harris_corners(
            gray
        )

        corner_result = draw_harris_corners(
            image,
            corners
        )

        save_image(
            os.path.join(
                args.output,
                "03_canny.jpg"
            ),
            edges
        )

        save_image(
            os.path.join(
                args.output,
                "04_log.jpg"
            ),
            log_result
        )

        save_image(
            os.path.join(
                args.output,
                "05_dog.jpg"
            ),
            dog_result
        )

        save_image(
            os.path.join(
                args.output,
                "06_hough.jpg"
            ),
            hough_result
        )

        save_image(
            os.path.join(
                args.output,
                "07_harris.jpg"
            ),
            corner_result
        )

        print(
            "    Canny: completed"
        )

        print(
            "    LoG: completed"
        )

        print(
            "    DoG: completed"
        )

        print(
            "    Hough: completed"
        )

        print(
            "    Harris: completed"
        )

    # ------------------------------------------------
    # FEATURE EXTRACTION
    # ------------------------------------------------

    keypoints, descriptors = (
        extract_sift_features(gray)
    )

    hog_features, hog_image = (
        extract_hog(gray)
    )

    gabor_responses = (
        gabor_features(gray)
    )

    dwt_components = haar_dwt(
        gray
    )

    if args.method in [
        "all",
        "features"
    ]:

        print(
            "\n[2] Extracting visual features..."
        )

        sift_result = draw_sift_features(
            image,
            keypoints
        )

        save_image(
            os.path.join(
                args.output,
                "08_sift.jpg"
            ),
            sift_result
        )

        save_hog_visualization(
            hog_image,
            os.path.join(
                args.output,
                "09_hog.jpg"
            )
        )

        for index, response in enumerate(
            gabor_responses
        ):

            save_image(
                os.path.join(
                    args.output,
                    f"10_gabor_{index}.jpg"
                ),
                response
            )

        for index, component in enumerate(
            dwt_components
        ):

            save_image(
                os.path.join(
                    args.output,
                    f"11_dwt_{index}.jpg"
                ),
                component
            )

        print(
            f"    SIFT keypoints: "
            f"{len(keypoints)}"
        )

        print(
            f"    HOG feature length: "
            f"{len(hog_features)}"
        )

        print(
            "    Gabor responses: completed"
        )

        print(
            "    Haar DWT: completed"
        )

    # ------------------------------------------------
    # SEGMENTATION
    # ------------------------------------------------

    threshold, otsu_mask = (
        otsu_segmentation(gray)
    )

    grabcut_mask = (
        grabcut_segmentation(image)
    )

    grabcut_result = apply_mask(
        image,
        grabcut_mask
    )

    growing_result = region_growing(
        gray
    )

    if args.method in [
        "all",
        "segmentation"
    ]:

        print(
            "\n[3] Running segmentation..."
        )

        save_image(
            os.path.join(
                args.output,
                "12_otsu.jpg"
            ),
            otsu_mask
        )

        save_image(
            os.path.join(
                args.output,
                "13_grabcut_mask.jpg"
            ),
            grabcut_mask
        )

        save_image(
            os.path.join(
                args.output,
                "14_grabcut_result.jpg"
            ),
            grabcut_result
        )

        save_image(
            os.path.join(
                args.output,
                "15_region_growing.jpg"
            ),
            growing_result
        )

        print(
            f"    Otsu threshold: "
            f"{threshold:.2f}"
        )

        print(
            "    GrabCut: completed"
        )

        print(
            "    Region growing: completed"
        )

    # ------------------------------------------------
    # STATISTICS
    # ------------------------------------------------

    statistics = calculate_feature_statistics(
        gray,
        edges,
        keypoints,
        hog_features,
        gabor_responses
    )

    foreground = foreground_percentage(
        grabcut_mask
    )

    save_summary(
        statistics,
        foreground,
        os.path.join(
            args.output,
            "summary.txt"
        )
    )

    print(
        "\n[4] Analysis summary"
    )

    print(
        f"    Edge pixels: "
        f"{statistics['edge_pixels']}"
    )

    print(
        f"    Edge percentage: "
        f"{statistics['edge_percentage']}%"
    )

    print(
        f"    SIFT keypoints: "
        f"{statistics['sift_keypoints']}"
    )

    print(
        f"    HOG dimensions: "
        f"{statistics['hog_feature_length']}"
    )

    print(
        f"    GrabCut foreground: "
        f"{foreground}%"
    )

    print(
        "\nProcessing completed."
    )

    print(
        f"Results saved in: "
        f"{args.output}"
    )

    print("=" * 55)


if __name__ == "__main__":
    main()