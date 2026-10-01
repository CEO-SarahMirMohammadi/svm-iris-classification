# SVM Iris Classification
# Repository #1
#
# This program:
# 1. Loads the Iris dataset from a CSV URL
# 2. Explores the dataset
# 3. Removes missing values
# 4. Plots sepal length vs. sepal width
# 5. Trains an SVM classifier
# 6. Predicts the species of a new flower

from sklearn import svm
import pandas as pd
from matplotlib import pyplot as plt


# ---------------------------------------------------------
# 1. Load the Iris dataset
# ---------------------------------------------------------

url = (
    "https://gist.githubusercontent.com/curran/"
    "a08a1080b88344b0c8a7/raw/"
    "0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
)

df = pd.read_csv(url)


# ---------------------------------------------------------
# 2. Explore the dataset
# ---------------------------------------------------------

print("Dataset shape:")
print(df.shape)

print("\nFirst 10 rows:")
print(df.head(10))

print("\nLast 10 rows:")
print(df.tail(10))

print("\nDataset statistics:")
print(df.describe())


# ---------------------------------------------------------
# 3. Check and remove missing values
# ---------------------------------------------------------

print("\nTotal number of missing values:")
print(df.isna().sum().sum())

df = df.dropna()


# ---------------------------------------------------------
# 4. Display the number of samples per species
# ---------------------------------------------------------

print("\nNumber of samples per species:")
print(df.groupby("species").size())


# ---------------------------------------------------------
# 5. Plot the first two features
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    df["sepal_length"],
    df["sepal_width"],
    alpha=0.7
)

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Iris Dataset - Sepal Length vs Sepal Width")
plt.grid(True)

plt.show()


# ---------------------------------------------------------
# 6. Prepare data for SVM
# ---------------------------------------------------------

# Use the first two features:
# Sepal Length and Sepal Width

X = df[["sepal_length", "sepal_width"]].values

# Get the target labels
species = df["species"]


# ---------------------------------------------------------
# 7. Convert species names to numerical labels
# ---------------------------------------------------------

label_mapping = {
    species_name: label
    for label, species_name
    in enumerate(sorted(set(species)))
}

y = [label_mapping[flower] for flower in species]

print("\nSpecies-to-label mapping:")
print(label_mapping)


# ---------------------------------------------------------
# 8. Train the SVM classifier
# ---------------------------------------------------------

clf = svm.SVC()

clf.fit(X, y)


# ---------------------------------------------------------
# 9. Make a prediction
# ---------------------------------------------------------

# Given flower:
# Sepal Length = 5.4 cm
# Sepal Width = 3.2 cm

new_flower = [[5.4, 3.2]]

prediction = clf.predict(new_flower)


print("\nPrediction:")
print(prediction)

print("\nPredicted species:")

# Convert numerical prediction back to species name
reverse_mapping = {
    value: key
    for key, value in label_mapping.items()
}

print(reverse_mapping[prediction[0]])
