import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

# Generate clustered data for testing using numpy and random
import random
def generate_test_data(num_clusters=5, output_file='test_detections.csv'):
    labels = ['object1', 'object2', 'object3']
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['label', 'x', 'y', 'z'])
        for _ in range(num_clusters):
            label = random.choice(labels)
            cluster_center = np.random.uniform(-5, 5, 3)  # Random cluster center
            for _ in range(random.randint(5, 10)):  # Random number of dete
                noise = np.random.normal(0, 0.2, 3)  # Small noise
                x, y, z = cluster_center + noise
                writer.writerow([label, x, y, z])


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def cluster_detections(input_csv, output_csv, threshold=1.0):
    detections = []
    with open(input_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            label = row[0]
            x, y, z = map(float, row[1:])
            detections.append({'pos': (x, y, z), 'label': label, 'row': row})

    clustered = []
    used = set()
    for i, det in enumerate(detections):
        if i in used:
            continue
        cluster = [det]
        used.add(i)
        for j, other in enumerate(detections):
            if j in used or det['label'] != other['label']:
                continue
            if euclidean_distance(det['pos'], other['pos']) <= threshold:
                cluster.append(other)
                used.add(j)
        # Average position, keep label
        avg_x = sum(d['pos'][0] for d in cluster) / len(cluster)
        avg_y = sum(d['pos'][1] for d in cluster) / len(cluster)
        avg_z = sum(d['pos'][2] for d in cluster) / len(cluster)
        clustered.append([det['label'], avg_x, avg_y, avg_z])

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for row in clustered:
            writer.writerow(row)

def plot_clusters(raw, processed):
    cluster_centers_estimated = []
    with open(processed, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        if header[0] != 'class':
            raise ValueError("Expected first column to be 'label' in clustered_detections.csv")
        cluster_centers_estimated = np.array([[float(value) for value in row[1:]] for row in reader])

    all_data = []
    with open(raw, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        if header[0] != 'class':
            raise ValueError("Expected first column to be 'label' in test_detections.csv")
        all_data = np.array([[float(value) for value in row[1:]] for row in reader])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.scatter(all_data[:, 0], all_data[:, 1], c='blue', label='All Detections', alpha=0.5)
    ax.scatter(cluster_centers_estimated[:, 0], cluster_centers_estimated[:, 1], c='red', label='Cluster Centers', s=100)
    plt.savefig('clusters_plot.png')

if __name__ == "__main__":
    cluster_detections(sys.argv[1], sys.argv[2])
    plot_clusters(sys.argv[1], sys.argv[2])

