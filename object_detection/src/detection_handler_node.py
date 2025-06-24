#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
from sensor_msgs_py import point_cloud2
import time
from os.path import join
from numpy.lib.recfunctions import unstructured_to_structured
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, PointField
from geometry_msgs.msg import PoseArray, Pose, Quaternion

from object_detection_msgs.msg import (
    PointCloudArray,
    ObjectDetectionInfo,
    ObjectDetectionInfoArray,
)
from std_msgs.msg import Header

from object_detection.utils import *

# from object_detection.ros_numpy import *

from ament_index_python.packages import get_package_share_directory


class DetectionHandlerNode(Node):
    def __init__(self):
        super().__init__("object_detection_node")

        self.get_logger().info(
            "[ObjectDetection Node] Object Detector initilization starts ..."
        )

        # ---------- Initialize parameters ----------
        self.declare_parameters(
            namespace="",
            parameters=[
                ("verbose", True),
                ("detections_info_topic", "/detection_info"),
            ],
        )

        # ---------- Setup subscribers ----------
        self.detections_sub = Subscriber(
            self,
            ObjectDetectionInfoArray,
            self.get_parameter("detections_info_topic").value,
        )

        self.get_logger().info(
            "[DetectionHandlerNode] Subscribing to detection info topic: "
            f"{self.get_parameter('detections_info_topic').value}"
        )

    def detection_info_callback(self, msg):
        """Handle detection info message"""
        if not self.image_info_received:
            self.get_logger().warn(
                "[ObjectDetection Node] Waiting for camera info...", once=True
            )
            return

        # Process detection info
        if msg.info:
            for info in msg.info:
                if info.class_id and info.id:
                    self.get_logger().info(
                        f"Detected object: {info.class_id} with ID {info.id} at position ({info.position.x}, {info.position.y}, {info.position.z})"
                    )


def main(args=None):
    rclpy.init(args=args)

    try:
        node = DetectionHandlerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down")
    except Exception as e:
        node.get_logger().fatal(f"Fatal error: {str(e)}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
