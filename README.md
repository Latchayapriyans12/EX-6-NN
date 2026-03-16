3>ENTER YOUR NAME:Latchaya priyan S</H3>
<H3>ENTER YOUR REGISTER NO.212224230139</H3>
<H3>EX. NO.6</H3>
<H3>DATE:</H3>
<H1 ALIGN =CENTER>Heart attack prediction using MLP</H1>
<H3>Aim:</H3>  To construct a  Multi-Layer Perceptron to predict heart attack using Python
<H3>Algorithm:</H3>
Step 1:Import the required libraries: numpy, pandas, MLPClassifier, train_test_split, StandardScaler, accuracy_score, and matplotlib.pyplot.<BR>
Step 2:Load the heart disease dataset from a file using pd.read_csv().<BR>
Step 3:Separate the features and labels from the dataset using data.iloc values for features (X) and data.iloc[:, -1].values for labels (y).<BR>
Step 4:Split the dataset into training and testing sets using train_test_split().<BR>
Step 5:Normalize the feature data using StandardScaler() to scale the features to have zero mean and unit variance.<BR>
Step 6:Create an MLPClassifier model with desired architecture and hyperparameters, such as hidden_layer_sizes, max_iter, and random_state.<BR>
Step 7:Train the MLP model on the training data using mlp.fit(X_train, y_train). The model adjusts its weights and biases iteratively to minimize the training loss.<BR>
Step 8:Make predictions on the testing set using mlp.predict(X_test).<BR>
Step 9:Evaluate the model's accuracy by comparing the predicted labels (y_pred) with the actual labels (y_test) using accuracy_score().<BR>
Step 10:Print the accuracy of the model.<BR>
Step 11:Plot the error convergence during training using plt.plot() and plt.show().<BR>
<H3>Program: </H3>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

# -----------------------------
# 1. Load and prepare dataset
# -----------------------------
data = pd.read_csv('heart.csv')

# Features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 2. Initialize MLP model
# -----------------------------
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # two hidden layers with 128 and 64 neurons
    activation='relu',             # ReLU activation
    solver='adam',                 # Adam optimizer
    alpha=0.0001,                  # L2 regularization
    batch_size='auto',
    learning_rate='adaptive',      # adaptive learning rate
    max_iter=1000,
    early_stopping=True,           # stops training if validation score doesn't improve
    n_iter_no_change=20,
    random_state=42,
    verbose=True
)

# -----------------------------
# 3. Train the model
# -----------------------------
history = mlp.fit(X_train, y_train)

# -----------------------------
# 4. Predictions and Evaluation
# -----------------------------
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on test set: {accuracy:.4f}\n")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification report
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# 5. Plot training loss curve
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(mlp.loss_curve_, label='Training Loss')
plt.title("MLP Training Loss Convergence")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()

# -----------------------------
# 6. Feature importance (Permutation Importance)
# -----------------------------
result = permutation_importance(mlp, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
importance = result.importances_mean
feature_names = data.columns[:-1]

plt.figure(figsize=(8,5))
sns.barplot(x=importance, y=feature_names)
plt.title("Feature Importance (Permutation)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

```

<H3>Output:</H3>
EX-6-NN/Screenshot 2026-03-16 142048.png
<H3>Results:</H3>
Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.
