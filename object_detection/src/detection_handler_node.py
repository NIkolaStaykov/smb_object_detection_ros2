#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from os.path import join
from numpy.lib.recfunctions import unstructured_to_structured
from message_filters import Subscriber
from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)

import tf2_ros
from tf2_geometry_msgs import do_transform_point

from geometry_msgs.msg import PointStamped


from object_detection_msgs.msg import (
    PointCloudArray,
    ObjectDetectionInfo,
    ObjectDetectionInfoArray,
)
from std_msgs.msg import Header

from object_detection.utils import *

from ament_index_python.packages import get_package_share_directory


class DetectionHandlerNode(Node):
    def __init__(self):
        super().__init__("detection_handler_node")

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
        self.detections_sub.registerCallback(self.detection_info_callback)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(
            "[DetectionHandlerNode] Subscribing to detection info topic: "
            f"{self.get_parameter('detections_info_topic').value}"
        )

        # ------------

    def detection_info_callback(self, msg):
        """Handle detection info message"""
        self.get_logger().info(
            f"Received {msg.header} object detections from topic "
            f"{self.get_parameter('detections_info_topic').value}"
        )
        for object_info in msg.info:
            self.get_logger().info(
                f"Object ID: {object_info.id}, "
                f"Class: {object_info.class_id}, "
                f"Confidence: {object_info.confidence}, "
                f"Position: {object_info.position}, "
            )

            if self.class_is_interesting(object_info.class_id):
                world_frame_point = self.transform_point(
                    object_info.position,
                    from_frame="rgb_camera_optical_link",
                    to_frame="map_o3d_graph_msf_aligned",
                    stamp=rclpy.time.Time(),
                )

                self.get_logger().info(
                    f"Transformed Position in 'map' frame: "
                    f"({world_frame_point.point.x}, "
                    f"{world_frame_point.point.y}, "
                    f"{world_frame_point.point.z})"
                )

    def class_is_interesting(self, class_id):
        # TODO: Implement later based on instructions
        return True

    def transform_point(self, point, from_frame, to_frame, stamp=None):
        """
        Transform a geometry_msgs/Point or PointStamped from from_frame to to_frame.
        """
        if not isinstance(point, PointStamped):
            ps = PointStamped()
            ps.header.frame_id = from_frame
            ps.header.stamp = stamp if stamp else self.get_clock().now().to_msg()
            ps.point.x = point.x
            ps.point.y = point.y
            ps.point.z = point.z
        else:
            ps = point

        try:
            # Wait for transform to be available
            trans = self.tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                rclpy.time.Time() if stamp is None else stamp,
                timeout=rclpy.duration.Duration(seconds=1.0),
            )

            # Transform the point
            self.get_logger().info(f"transform: {trans.transform}, ")

            transformed = do_transform_point(ps, trans)
            return transformed

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF transform failed: {e}")
            return None


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
