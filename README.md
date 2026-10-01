# SVM Iris Classification

A simple machine learning project that uses a Support Vector Machine (SVM) to classify Iris flowers based on their sepal length and sepal width.

The project also visualizes the first two features of the Iris dataset using a scatter plot.

## Overview

This project demonstrates a basic machine learning workflow:

1. Load a dataset from a CSV URL
2. Explore the dataset
3. Check for missing values
4. Remove missing values
5. Visualize two features using a scatter plot
6. Prepare features and target labels
7. Convert categorical labels into numerical values
8. Train a Support Vector Machine classifier
9. Make a prediction for a new data point

## Algorithm

### Support Vector Machine

Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks.

For classification, SVM attempts to find a decision boundary that separates different classes while maximizing the margin between the classes.

In this project, the SVM classifier uses:

- Sepal Length
- Sepal Width

as input features.

The target variable is the Iris flower species.

## Dataset

The project uses the Iris dataset.

The dataset contains measurements of Iris flowers from three different species:

- Iris setosa
- Iris versicolor
- Iris virginica

Each sample contains four numerical measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

However, this project intentionally uses only the first two features:

- Sepal length
- Sepal width

## Visualization

The project creates a scatter plot using:

- X-axis: Sepal Length
- Y-axis: Sepal Width

Each point represents one flower in the dataset.

## Example Prediction

The trained SVM model is used to predict the species of a flower with:

```text
Sepal Length = 5.4 cm
Sepal Width = 3.2 cm
