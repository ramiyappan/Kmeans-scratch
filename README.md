# Implementing KMeans from scratch

## Files on the Repository

  `Image` - *Folder that contains Dataset and Main Program for Clustering Hand-written digits.*

  `Iris`  - *Folder that contains Dataset and Main Program for Clustering Flowers.*
  
  `HW4_IYAPPAN.zip` - *is the submission zip file for the homework.*
  
  `K Means Clustering Report.pdf` - *is the submission file for the final report.*

## Business Problem

Use K-Means Clustering Algorithm to:
  - (1) Predict the type of flower, and, 
  - (2) Deal with Image data (pixels) to predict hand-written digits (0-9).

## Objectives

* Implement the K-Means Clustering algorithm.
* Deal with image data (processed and stored in vector format).
* Explore methods for dimensionality reduction.
* Think about metrics for evaluating Clustering solutions.

## Dataset

### Flowers

Use the famous Iris dataset which consists of 150 instances & 4 features:

  - sepal length in cm
  - sepal width in cm
  - petal length in cm
  - petal width in cm

### Hand-written Digits

Input Data consists of 10,000 images of handwritten digits (0-9). The images were scanned and scaled into 28x28 pixels. For every digit, each pixel can be represented as an integer in the range [0, 255] where 0 corresponds to the pixel being completely white, and 255 corresponds to the pixel being completely black. This gives us a 28x28 matrix of integers for each digit. We can then flatten each matrix into a 1x784 vector. No labels are provided.

Format of the input data: Each row is a record (image), which contains 784 comma-delimited integers.

Examples of digit images can be found at http://cs.gmu.edu/~sanmay/ImageExamples.png

## Task 1 - Clustering Flowers

Test K-means Algorithm on the easy and famous Iris dataset. This will serve as a benchmark for task-2, making it easier to understand about the algorithm using this simple dataset in task-1.

Assign the 150 instances in the test file to 3 cluster ids given by 1, 2 or 3.

## Task 2 - Clustering Digits

Essentially your task is to assign each of the instances in the input data to K clusters identified from 1 to K. So, used K=10 Clusters for classifying the digits from 0-9.

format.txt shows an example file containing 10,000 rows with random class assignments from 1 to 10.

For this specific dataset and model, t-SNE works better as a Dimensionality Reduction technique. But anyways first started to explore PCA, so continued with that and the model hardly performed very average using PCA. You can also use other techniques like feature selection, SVD, etc.


