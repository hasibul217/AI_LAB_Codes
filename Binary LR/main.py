import numpy as np
import csv
import matplotlib.pyplot as plt

# Load dataset diabetes
data = []
with open('diabetes.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data.append([float(x) for x in row])

data = np.array(data)
np.random.shuffle(data)

# Split the dataset into Training, Validation, and Test sets
train_size = int(0.7 * len(data))
val_size = int(0.15 * len(data))
test_size = len(data) - (train_size + val_size)
train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]


# Feature Scaling (Manually)
def scale_features(dataset):
    scaled_dataset = dataset.copy()
    num_features = dataset.shape[1] - 1  # Exclude the target column
    for i in range(num_features):
        feature_col = dataset[:, i]
        mean = np.mean(feature_col)
        std = np.std(feature_col)
        scaled_dataset[:, i] = (feature_col - mean) / std
    return scaled_dataset


train_data[:, :-1] = scale_features(train_data[:, :-1])
val_data[:, :-1] = scale_features(val_data[:, :-1])
test_data[:, :-1] = scale_features(test_data[:, :-1])

# Initialize parameters
theta = np.random.rand(train_data.shape[1])
lr_values = [0.1,0.01, 0.001, 0.0001]
max_iter = 500

# Training
train_loss = []
for lr in lr_values:
    theta = np.random.rand(train_data.shape[1])
    lr_history = []
    for itr in range(1, max_iter + 1):
        TJ = 0
        dv_sum = np.zeros(train_data.shape[1])
        for sample in train_data:
            X = np.concatenate((sample[:-1], [1]))
            z = np.dot(X, theta)
            h = 1 / (1 + np.exp(-z))
            J = -sample[-1] * np.log1p(h) - (1 - sample[-1]) * np.log1p(1 - h)
            TJ += J
            dv_sum += X * (h - sample[-1])
        TJ /= len(train_data)
        dv_sum /= len(train_data)
        lr_history.append(TJ)
        theta -= lr * dv_sum
    train_loss.append(lr_history)

# Calculate validation accuracy
val_accs = []
for lr, lr_history in zip(lr_values, train_loss):
    correct = 0
    for sample in val_data:
        X = np.concatenate((sample[:-1], [1]))
        z = np.dot(X, theta)
        h = 1 / (1 + np.exp(-z))
        if h >= 0.5:
            h = 1
        else:
            h = 0
        if h == sample[-1]:
            correct += 1
    val_acc = (correct / len(val_data)) * 100
    val_accs.append(val_acc)

# Find learning rate with maximum validation accuracy
best_lr_idx = np.argmax(val_accs)
best_lr = lr_values[best_lr_idx]

# Calculate test accuracy
correct = 0
for sample in test_data:
    X = np.concatenate((sample[:-1], [1]))
    z = np.dot(X, theta)
    h = 1 / (1 + np.exp(-z))
    if h >= 0.5:
        h = 1
    else:
        h = 0
    if h == sample[-1]:
        correct += 1
test_acc = (correct / len(test_data)) * 100

# Plot train loss vs. iteration
iterations = range(1, max_iter + 1)
for lr, lr_history in zip(lr_values, train_loss):
    plt.plot(iterations, lr_history, label=f'lr = {lr}')
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. Iteration')
plt.legend()
plt.show()

# Print


# Print validation and test accuracy
print("Validation Accuracies:", val_accs)
print("Best Learning Rate:", best_lr)
print("Test Accuracy with Best LR:", test_acc)
