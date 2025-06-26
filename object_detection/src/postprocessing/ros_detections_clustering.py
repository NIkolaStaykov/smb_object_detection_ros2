import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def cluster_detections(detections, distance_threshold=1.5):
    """
    Clusters detections based on a distance threshold.
    
    Args:
        detections (pd.DataFrame): DataFrame containing detection data with 'x', 'y', and 'z' columns.
        distance_threshold (float): Maximum distance to consider detections as part of the same cluster.
        
    Returns:
        pd.DataFrame: DataFrame with clustered detections.
    """
    if detections.empty:
        return pd.DataFrame(columns=detections.columns)

    # Calculate pairwise distances
    cols = ['pose.position.x', 'pose.position.y', 'pose.position.z']
    coords = detections[cols].values
    print("Coordinates shape:", coords.shape)
    distances = np.linalg.norm(coords[:, np.newaxis] - coords, axis=-1)

    # Create clusters based on distance threshold
    clusters = []
    visited = set()
    
    for i in range(len(coords)):
        if i in visited:
            continue
        
        cluster = [i]
        visited.add(i)
        
        for j in range(i + 1, len(coords)):
            if j not in visited and distances[i, j] < distance_threshold:
                cluster.append(j)
                visited.add(j)
        
        clusters.append(cluster)

    # Create a new DataFrame for clustered detections
    clustered_detections = []
    
    for cluster in clusters:
        cluster_data = detections[cols].iloc[cluster].mean().to_dict()
        clustered_detections.append(cluster_data)
    
    return pd.DataFrame(clustered_detections)


def plot_clusters(detections, clusters):
    """
    Plots the clustered detections.
    
    Args:
        detections (pd.DataFrame): Original detection DataFrame.
        clusters (pd.DataFrame): Clustered detection DataFrame.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot original detections
    ax.scatter(detections['pose.position.x'], detections['pose.position.y'], c='r', label='Original Detections')

    # Plot clustered detections
    ax.scatter(clusters['pose.position.x'], clusters['pose.position.y'], c='b', label='Clustered Detections', marker='x')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()
    
    plt.savefig('clusters.png')

if __name__ == "__main__":
    print(sys.argv[1])
    detections = pd.read_csv(sys.argv[1])
    print("Loaded detections from:", sys.argv[0])
    clusters = cluster_detections(detections)
    plot_clusters(detections, clusters)