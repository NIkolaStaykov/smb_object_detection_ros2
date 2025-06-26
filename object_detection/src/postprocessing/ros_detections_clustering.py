import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
import csv


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

    bag_path     = str(sys.argv[1])
    topic_filter = "/detection_position_transformed"

    # Configure storage and conversion options
    storage_options = StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # Get topic types
    topic_types = reader.get_all_topics_and_types()
    type_dict = {topic.name: topic.type for topic in topic_types}

    msgs = []

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic == topic_filter:
            msg_type = get_message(type_dict[topic])
            msg = deserialize_message(data, msg_type)
            msgs.append((t, msg))

    csv_path = "detections_j.csv"

    if msgs:
        # Flattened CSV columns: timestamp, header.stamp.sec, header.stamp.nanosec, header.frame_id, pose.position.x, pose.position.y, pose.position.z, pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        fields = [
            'timestamp',
            'header.stamp.sec', 'header.stamp.nanosec', 'header.frame_id',
            'pose.position.x', 'pose.position.y', 'pose.position.z',
            'pose.orientation.x', 'pose.orientation.y', 'pose.orientation.z', 'pose.orientation.w'
        ]
        with open(csv_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(fields)
            for t, m in msgs:
                # Flatten header
                header = m._header
                stamp = header.stamp
                frame_id = header.frame_id
                # Each pose in _poses
                for pose in m._poses:
                    pos = pose.position
                    ori = pose.orientation
                    row = [
                        t,
                        stamp.sec, stamp.nanosec, frame_id,
                        pos.x, pos.y, pos.z,
                        ori.x, ori.y, ori.z, ori.w
                    ]
                    writer.writerow(row)


    detections = pd.read_csv(csv_path)
    print(detections)
    print("Loaded detections from:", sys.argv[0])
    clusters = cluster_detections(detections)
    clusters.to_csv("clustered_objects_j.csv")
    # plot_clusters(detections, clusters)