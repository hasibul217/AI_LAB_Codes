import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the Dataset
df = pd.read_csv('dataset.csv')

# Step 2: Data Preprocessing
# Normalize the numerical columns
numerical_cols = ['Assignment-1', 'Assignment-2', 'Assignment-3', 'Assignment-4', 'Assignment-5', 'Final', 'Mid']
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Convert 'Section' column to numerical values
df['Section'] = df['Section'].map({'A': 0, 'B': 1})

# Step 3: Split the Dataset
X = df.drop(['Roll', 'Section'], axis=1)
y = df['Section']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Step 4: Train and Tune the KNN Model
best_accuracy = 0
best_k = 0

# Define a range of K values to test
k_values = [1, 3, 5, 7, 9]

for k in k_values:
    # Train a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Evaluate on the validation set
    y_val_pred = knn.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)

    # Check if the current K value gives better accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

# Step 5: Test the Model
# Train the KNN model using the best K value on the combined training and validation sets
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

# Evaluate on the testing set
y_test_pred = knn_best.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Step 6: Interpret the Results
print(f"Best K value: {best_k}")
print(f"Validation Accuracy: {best_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")
