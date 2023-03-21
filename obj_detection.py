import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dataset
data = io.imread_collection('Data/*.jpg')

# Convert images to features
X = []
for image in data:
    # Convert to grayscale and resize to 100x100
    image = rgb2gray(resize(image, (100, 100)))
    # Flatten image into 1D array
    features = image.ravel()
    X.append(features)
X = np.array(X)

# Create labels for each image
y = []
for i in range(len(data)):
    if i < 50:
        y.append(0) # label as category 0
    else:
        y.append(1) # label as category 1
y = np.array(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train logistic regression model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Make predictions on test set
y_pred = clf.predict(X_test)

# Calculate accuracy of model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
