# MNIST Digit Visualization with OpenCV and Python

## Project Overview

This project focuses on exploring and visualizing the MNIST handwritten-digit dataset using Python and computer vision libraries.

The MNIST dataset contains grayscale images of handwritten digits from 0 to 9. Each image is represented by 784 pixel values arranged in a 28 × 28 matrix.

In this project, we load the training and testing datasets from CSV files, separate the digit labels from pixel data, convert individual samples into 2D image arrays, and visualize them using Matplotlib.

The primary objective is to understand how image data is represented numerically and how it can be converted into a visual format for further machine-learning applications.

## Project Objectives

* Load MNIST training and testing datasets using pandas.
* Understand the structure of the MNIST dataset.
* Separate independent features (pixel values) and dependent labels (digit classes).
* Convert 784 pixel values into a 28 × 28 image.
* Visualize handwritten digits using Matplotlib.
* Perform exploratory data analysis (EDA) to understand the dataset.
* Prepare the dataset for future machine-learning classification.

## Dataset Description

The project uses two CSV files:

* `mnist_train.csv` – Training dataset.
* `mnist_test.csv` – Testing dataset.

Each row represents one handwritten digit.

| Feature               | Description            |
| --------------------- | ---------------------- |
| First column          | Digit label (0–9)      |
| Remaining 784 columns | Pixel intensity values |
| Image dimensions      | 28 × 28 pixels         |
| Image format          | Grayscale              |

Pixel values typically range from 0 to 255, representing different grayscale intensities.

## Project Structure

```text
Navie_Bayes_Mnist_opencv_project/
│
├── main.py
├── Minist_train_explanation_using_CV.py
├── Minist_test_explanation_Using_CV.py
├── mnist_train.csv
├── mnist_test.csv
├── requirements.txt
└── README.md
```

## Technologies Used

* **Python** – Main programming language.
* **pandas** – Loading and handling CSV datasets.
* **NumPy** – Converting and reshaping pixel data.
* **Matplotlib** – Visualizing handwritten digit images.
* **OpenCV** – Computer vision library used in the project environment.
* **scikit-learn** – Available for future machine-learning extensions.

## Project Workflow

1. Load the MNIST CSV dataset using pandas.
2. Separate the pixel features (`X`) and digit labels (`y`).
3. Select an individual image sample.
4. Convert the selected row into a NumPy array.
5. Reshape the 784 pixel values into a 28 × 28 matrix.
6. Display the image using Matplotlib.
7. Explore the dataset using EDA techniques.

### Example: Visualizing a Digit

```python
image = X.iloc[5, :].to_numpy().reshape(28, 28)

plt.imshow(image, cmap="gray")
plt.axis("off")
plt.show()
```

## Exploratory Data Analysis (EDA)

EDA helps us understand the dataset's structure, quality, distribution, and visual characteristics before applying machine-learning algorithms.

### 1. Dataset Dimensions and Preview

```python
print("Training dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)

print(train_df.head())
train_df.info()
```

**Important points:**

* Check the number of rows and columns.
* Verify that there is one label column and 784 pixel columns.
* Inspect sample records and data types.

### 2. Missing Values Analysis

```python
print("Training missing values:", train_df.isna().sum().sum())
print("Testing missing values:", test_df.isna().sum().sum())
```

**Important points:**

* Identify missing labels or pixel values.
* Confirm whether preprocessing is required before modeling.

### 3. Digit Label Distribution

```python
import matplotlib.pyplot as plt

train_df.iloc[:, 0].value_counts().sort_index().plot(kind="bar")

plt.title("MNIST Training Label Distribution")
plt.xlabel("Digit")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()
```

**Important points:**

* Understand how many examples are available for each digit.
* Check whether the dataset is balanced across classes.
* Identify digits with fewer training examples.

### 4. Pixel Intensity Distribution

```python
pixels = train_df.iloc[:, 1:].to_numpy()

plt.hist(pixels.ravel(), bins=30)

plt.title("Pixel Intensity Distribution")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
```

**Important points:**

* Understand the distribution of grayscale pixel values.
* Identify whether most pixels are dark or bright.
* Learn how numerical pixel values represent image information.

### 5. Visualizing Sample Digits

```python
fig, axes = plt.subplots(2, 5, figsize=(10, 4))

for digit, ax in enumerate(axes.ravel()):
    row = train_df[train_df.iloc[:, 0] == digit].iloc[0, 1:]

    ax.imshow(row.to_numpy().reshape(28, 28), cmap="gray")
    ax.set_title(f"Label: {digit}")
    ax.axis("off")

plt.tight_layout()
plt.show()
```

**Important points:**

* Visualize examples of all ten digit classes.
* Observe handwriting differences between samples.
* Understand how digit shapes are represented by pixel patterns.

### 6. Average Image of a Digit

```python
digit = 5

digit_pixels = train_df[train_df.iloc[:, 0] == digit].iloc[:, 1:]

mean_image = digit_pixels.mean(axis=0).to_numpy().reshape(28, 28)

plt.imshow(mean_image, cmap="gray")
plt.title(f"Average Image of Digit {digit}")
plt.axis("off")
plt.show()
```

**Important points:**

* Observe the common pixel patterns for a selected digit.
* Understand the typical shape of that digit across multiple samples.
* Compare average images of different digit classes.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd Navie_Bayes_Mnist_opencv_project
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

### 3. Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the Project

```bash
python Minist_train_explanation_using_CV.py
```

```bash
python Minist_test_explanation_Using_CV.py
```

Run the scripts from the project root directory so the CSV file paths resolve correctly.

## Key Learnings

* Understanding the structure of image datasets stored in CSV format.
* Separating target labels from image pixel features.
* Converting one-dimensional pixel data into two-dimensional images.
* Visualizing grayscale images using Matplotlib.
* Performing basic EDA on image datasets.
* Preparing image data for future classification tasks.

## Future Improvements

* Create reusable functions for loading and displaying MNIST images.
* Add more EDA visualizations and statistical summaries.
* Implement a Naive Bayes classifier for digit recognition.
* Evaluate classification performance using accuracy, confusion matrix, and classification report.
* Develop a Flask web application to visualize or predict handwritten digits.
