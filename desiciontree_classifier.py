import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = datasets.load_iris(as_frame=False)
X = iris.data[:, 2:4]  # we only take the third and fourth features.
y = iris.target
feature_names = np.array(iris.feature_names)
class_names = iris.target_names

# Streamlit app title
st.title("Decision Tree Classifier for Iris Dataset")

# Display the scatter plot of the dataset
st.subheader("Iris Dataset Scatter Plot")
fig, ax = plt.subplots()
plt.xlabel(f"{feature_names[2]}")
plt.ylabel(f"{feature_names[3]}")
plt.title("Iris Dataset")
colors = ["navy", "turquoise", "darkorange"]
for color, i, target_name in zip(colors, [0, 1, 2], class_names):
    plt.scatter(
        X[y == i, 0], X[y == i, 1], color=color, alpha=0.8, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
st.pyplot(fig)

# Parameters for Decision Tree
st.sidebar.header("Model Parameters")
max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)

# Split the dataset
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train the Decision Tree model
DecisionTree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
DecisionTree.fit(X_train, y_train)

# Display training and testing accuracy
accuracy_score_train = DecisionTree.score(X_train, y_train)
accuracy_score_test = DecisionTree.score(X_test, y_test)

st.subheader("Model Performance")
st.write(f"Training Accuracy: {accuracy_score_train * 100:.2f}%")
misclassification_error_train = ceil((1 - accuracy_score_train) * len(y_test))
st.write(f"Misclassification Error on Training Dataset: {misclassification_error_train}")

st.write(f"Testing Accuracy: {accuracy_score_test * 100:.2f}%")
misclassification_error_test = ceil((1 - accuracy_score_test) * len(y_test))
st.write(f"Misclassification Error on Testing Dataset: {misclassification_error_test}")

# Display the trained decision tree
st.subheader("Trained Decision Tree")
fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(DecisionTree, feature_names=feature_names[2:4], class_names=class_names, filled=True)
plt.title("Trained Decision Tree")
st.pyplot(fig)
