# Predicting Type of Flower & Hand-written digits (0-9)

This project was part of assignments from the Data Mining course *(CS 584 - Spring '22)* at George Mason University.

## Files on the Repository

  Dataset ----------------> *is a folder that contains the train, test and format of output files for the dataset.*

  Graphs  -----------------> *is a folder that consists of 3 graphs with Number of iterations Vs the Cost.*

  GD Main.ipynb  --------> *is the main program in jupyter notebook.*
  
  GD Report.pdf ---------> *is the submission file for the final report.*

  GD.py ------------------> *is just the copy of whole program in python file.*

  HW2_IYAPPAN.zip -----> *is the submission zip file for the homework.*

## Business Problem

To predict whether a patient has heart disease or not by analyzing various features (BP, BodyFat, etc.,) using a Machine learning approach *(Logistic Regression using Gradient Descent in this case)*.

## Objectives

* Implement the K-Means algorithm.
* Deal with image data (processed and stored in vector format).
* Explore methods for dimensionality reduction.
* Think about metrics for evaluating clustering solutions.

## Dataset

Detailed Description:

There are 2 leaderboard submissions for this homework (they appear as separate parts on Miner). Iris is an easier dataset where K-Means can be tested quickly. The second part involves dealing with image data.

For this assignment, you are required to implement the K-Means algorithm. Please do not use libraries for implementing K-means. You may use them for preprocessing and dimensionality reduction.
Part 1 (Homework 5a on Miner)
The famous Iris dataset serves as an easy benchmark for evaluation. Test your K-Means Algorithm on this easy dataset with 4 features:

sepal length in cm
sepal width in cm
petal length in cm
petal width in cm
and 150 instances.

Assign the 150 instances in the test file to 3 cluster ids given by 1, 2 or 3. The leaderboard will output the V-measure and this benchmark can be used as an easy step for Part 2.

The training data is a NULL FILE.

The file iris_new_data.txt, under "Test data," contains the data you use for clustering.

The format example is given by iris_format.txt.

Part 2 (Homework 5b on Miner)
Input Data (provided as new_test.txt) consists of 10,000 images of handwritten digits (0-9). The images were scanned and scaled into 28x28 pixels. For every digit, each pixel can be represented as an integer in the range [0, 255] where 0 corresponds to the pixel being completely white, and 255 corresponds to the pixel being completely black. This gives us a 28x28 matrix of integers for each digit. We can then flatten each matrix into a 1x784 vector. No labels are provided.

Format of the input data: Each row is a record (image), which contains 784 comma-delimited integers.
Examples of digit images can be found at http://cs.gmu.edu/~sanmay/ImageExamples.png

For evaluation purposes (Leaderboard Ranking), we will use the V-measure in the sci-kit learn library that is considered an external index metric to evaluate clustering. Essentially your task is to assign each of the instances in the input data to K clusters identified from 1 to K.

For the leaderboard evaluation set K to 10. Submit your best results. The leaderboard will report the V-measure on 50% of sampled dataset.

Some things to note:

-- format.txt shows an example file containing 10,000 rows with random class assignments from 1 to 10.

-- You will almost certainly want to experiment with different dimensionality reduction techniques (e.g. feature selection, PCA, SVD, t-SNE), especially for Part 2.



