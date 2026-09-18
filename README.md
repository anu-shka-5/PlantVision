# PlantVision: Classical Computer Vision Based Plant Image Analysis

## 1. Project Overview

PlantVision is a command-line computer vision project for analyzing plant images using classical image processing and feature extraction techniques.

The project takes an input plant or flower image and performs preprocessing, edge detection, feature extraction, image segmentation, and basic visual analysis.

The implementation is based on concepts studied in the Computer Vision course, including Canny edge detection, Laplacian of Gaussian, Difference of Gaussian, Hough transform, Harris corner detection, SIFT, HOG, Gabor filters, Haar wavelet decomposition, Otsu thresholding, GrabCut, and region growing.

## 2. Objectives

The main objectives are:

* Preprocess an input plant image.
* Detect important image edges and structural features.
* Extract shape and local visual features.
* Analyze texture using Gabor filters and wavelet decomposition.
* Compare different segmentation techniques.
* Produce numerical feature statistics.
* Provide all processing through a command-line interface.

## 3. Technologies Used

* Python
* OpenCV
* NumPy
* scikit-image
* Matplotlib
* PyWavelets

## 4. Project Structure

```text
PlantVision/
├── README.md
├── requirements.txt
├── main.py
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── edges.py
│   ├── features.py
│   ├── segmentation.py
│   └── visualization.py
├── data/
│   └── sample.jpg
├── results/
├── notebooks/
│   └── unit-3-computer-vision.ipynb
└── report/
    └── PlantVision_Report.pdf
```

## 5. Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/anu-shka-5/PlantVision.git
cd PlantVision
```

### Step 2: Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 6. Running the Project

Place an input image inside the `data` directory.

For example:

```text
data/sample.jpg
```

Run the complete pipeline:

```bash
python main.py --input data/sample.jpg
```

## 7. Running Individual Modules

Run only edge-processing methods:

```bash
python main.py --input data/sample.jpg --method edges
```

Run feature extraction:

```bash
python main.py --input data/sample.jpg --method features
```

Run segmentation:

```bash
python main.py --input data/sample.jpg --method segmentation
```

Run the complete project:

```bash
python main.py --input data/sample.jpg --method all
```

## 8. Processing Pipeline

```text
Input Image
     |
     v
Preprocessing
     |
     +------------------+
     |                  |
     v                  v
Edge Detection     Feature Extraction
     |                  |
     |              SIFT / HOG
     |              Gabor / DWT
     |
     v
Segmentation
     |
 Otsu / GrabCut /
 Region Growing
     |
     v
Numerical Analysis
     |
     v
Saved Results
```

## 9. Computer Vision Techniques

### Edge Detection

The project implements:

* Canny
* Laplacian of Gaussian
* Difference of Gaussian

### Structural Feature Detection

The project uses:

* Probabilistic Hough transform
* Harris corner detection

### Feature Extraction

The project extracts:

* SIFT keypoints
* HOG descriptors
* Gabor texture responses
* Haar wavelet components

### Segmentation

The project includes:

* Otsu thresholding
* GrabCut
* Region growing

## 10. Output

The processed images and numerical analysis are saved in the `results` directory.

Typical outputs include:

```text
01_original.jpg
02_grayscale.jpg
03_canny.jpg
04_log.jpg
05_dog.jpg
06_hough.jpg
07_harris.jpg
08_sift.jpg
09_hog.jpg
10_gabor_0.jpg
10_gabor_1.jpg
10_gabor_2.jpg
10_gabor_3.jpg
11_dwt_0.jpg
11_dwt_1.jpg
11_dwt_2.jpg
11_dwt_3.jpg
12_otsu.jpg
13_grabcut_mask.jpg
14_grabcut_result.jpg
15_region_growing.jpg
summary.txt
```

## 11. Numerical Analysis

The generated summary includes measurements such as:

* Image dimensions
* Number of edge pixels
* Percentage of edge pixels
* Number of SIFT keypoints
* HOG feature-vector length
* Number of Gabor responses
* GrabCut foreground percentage

The values depend on the input image and are generated during execution.

## 12. Limitations

The project is based on classical computer vision techniques. The segmentation methods can be affected by image quality, background complexity, illumination, and object-background similarity.

The system does not claim to perform plant disease classification or semantic plant recognition. Its purpose is visual feature extraction, segmentation, and image analysis.

## 13. Future Scope

Possible future improvements include:

* Testing on a larger plant image dataset.
* Adding a supervised plant classification model.
* Comparing classical features with CNN-based features.
* Improving automatic foreground detection.
* Adding quantitative evaluation using manually labelled segmentation masks.
* Developing a simple graphical interface after the command-line version is established.

## 14. License

This project is developed as an academic Computer Vision course project.
