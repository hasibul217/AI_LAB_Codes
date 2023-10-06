import random
import matplotlib.pyplot as plt

import csv

# Read the content of the text file and create a CSV
with open('jain_feats.txt', 'r') as text_file:
    lines = text_file.readlines()

# Create a new CSV file
with open('jain_feats.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    for line in lines:
        data = line.strip()
        data_parts = data.split()

        if len(data_parts) == 2:
            csv_writer.writerow([data_parts[0], data_parts[1]])
        else:
            csv_writer.writerow([data_parts[0]])

    # Read the content of the text file and create a CSV
    with open('jain_centers.txt', 'r') as text_file:
        lines = text_file.readlines()

    # Create a new CSV file
    with open('jain_centers.csv', 'w', newline='') as csv_file2:
        csv_writer = csv.writer(csv_file2)

        for line in lines:
            data = line.strip()
            data_parts = data.split()

            if len(data_parts) == 2:
                csv_writer.writerow([data_parts[0], data_parts[1]])
            else:
                csv_writer.writerow([data_parts[0]])

my_id = 11201217
random.seed(my_id)
print(random.random())

Data = []
# Read data points from CSV file
with open('jain_feats.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        data = [float(val) for val in row]
        Data.append(data)

given_centers = []

# Read centers from CSV file
with open('jain_centers.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        center = [float(val) for val in row]
        given_centers.append(center)

# generating new Cluster Centers
for _ in range(1):
    new_center = []
    for dimension_values in zip(*Data):
        min_value = min(dimension_values)
        max_value = max(dimension_values)

        random_value = random.uniform(min_value, max_value)

        new_center.append(random_value)

    given_centers.append(new_center)

K_values = [3]


Clusters_list = []
for K in K_values:
    Centers = list(given_centers)

    Clusters = []
    for _ in range(K):
        Clusters.append([])

    itr, Shift = 1, 0
    max_iterations = 100

    while True:
        [cluster.clear() for cluster in Clusters]

        # add data points to clusters
        def euclidean_distance(a, b):
            return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

        for S in Data:
            closest_center = min(enumerate(Centers), key=lambda x: euclidean_distance(S, x[1]))[0]
            Clusters[closest_center].append(S)

        # Updating centers
        for i, cluster in enumerate(Clusters):
            if cluster:
                Centers[i] = [sum(dim) / len(cluster) for dim in zip(*cluster)]

        if itr > 1 and Shift < 50:
            break

        # Calculate shift
        Temp_Clusters = [[] for _ in range(K)]

        for S in Data:
            distances = [sum((a - b) ** 2 for a, b in zip(S, C)) ** 0.5 for C in Centers]
            closest_center = distances.index(min(distances))
            Temp_Clusters[closest_center].append(S)

        # Calculate shift
        Shift = 0

        for i in range(K):
            for S in Temp_Clusters[i]:
                if S not in Clusters[i]:
                    Shift += 1

        Clusters = Temp_Clusters
        itr += 1

        if itr <= max_iterations:
            pass
        else:
            break

    # Calculate inertia
    inertia = sum(sum((a - b) ** 2 for a, b in zip(S, Centers[i]))for cluster in Clusters for S in cluster)

    # Plot the data and clusters

    colors = ['seagreen', 'darkorchid', 'crimson', 'teal', 'coral', 'goldenrod']

    plt.figure(figsize=(8, 5))  # Create a new figure with a specific size

    for i, cluster in enumerate(Clusters):
        color = colors[i % len(colors)]
        points = [point for point in cluster]
        plt.scatter(*zip(*points), color=color, label=f'Cluster {i + 1}')

    plt.scatter(*zip(*Centers), color='black', marker='x', label='Centers')
    plt.title(f'Clustering with K = {K}, Inertia = {inertia:.2f}')
    plt.legend()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()

