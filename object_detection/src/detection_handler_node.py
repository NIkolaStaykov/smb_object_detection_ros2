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

from geometry_msgs.msg import PointStamped, PoseArray, Pose, Quaternion


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
                ("detection_point_clouds_topic", "/detection_point_clouds"),
                (
                    "detection_pointcloud_transformed_topic",
                    "/detection_pointcloud_transformed",
                ),
                (
                    "detection_position_transformed_topic",
                    "/detection_position_transformed",
                ),
            ],
        )

        self.map_frame_id = "map"
        self.camera_frame_id = "rgb_camera_optical_link"

        # ---------- Setup subscribers ----------
        self.detections_info_sub = Subscriber(
            self,
            ObjectDetectionInfoArray,
            self.get_parameter("detections_info_topic").value,
        )
        self.detections_info_sub.registerCallback(self.detection_info_callback)

        self.detections_pointcloud_sub = Subscriber(
            self,
            PointCloudArray,
            self.get_parameter("detection_point_clouds_topic").value,
        )
        self.detections_pointcloud_sub.registerCallback(
            self.detection_pointcloud_callback
        )

        # ------------ Setup publishers ------------
        self.detection_pointcloud_transformed_pub = self.create_publisher(
            PointCloud2,
            self.get_parameter("detection_pointcloud_transformed_topic").value,
            10,
        )

        self.detection_position_transformed_pub = self.create_publisher(
            PoseArray,
            self.get_parameter("detection_position_transformed_topic").value,
            10,
        )

        # ---------- Setup TF listener ----------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(
            "[DetectionHandlerNode] Subscribing to detection info topic: "
            f"{self.get_parameter('detections_info_topic').value}"
        )

    def detection_info_callback(self, msg):
        """Handle detection info message"""

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = self.map_frame_id

        world_frame_object_pose_array = PoseArray(header=header)

        for object_info in msg.info:
            # self.get_logger().info(
            #     f"Object ID: {object_info.id}, "
            #     f"Class: {object_info.class_id}, "
            #     f"Confidence: {object_info.confidence}, "
            #     f"Position: {object_info.position}, "
            #     f"Frame: {msg.header.frame_id}"
            # )

            if self.class_is_interesting(object_info.class_id):
                world_frame_point = self.transform_point(
                    object_info.position,
                    from_frame=self.camera_frame_id,
                    to_frame=self.map_frame_id,
                    stamp=rclpy.time.Time(),
                )

                world_frame_object_pose = Pose()
                world_frame_object_pose.position.x = float(world_frame_point.point.x)
                world_frame_object_pose.position.y = float(world_frame_point.point.y)
                world_frame_object_pose.position.z = float(world_frame_point.point.z)
                world_frame_object_pose.orientation = Quaternion(
                    x=0.0, y=0.0, z=0.0, w=1.0
                )
                world_frame_object_pose_array.poses.append(world_frame_object_pose)

                # camera frame pose
                camera_frame_pose = Pose()
                camera_frame_pose.position = object_info.position
                camera_frame_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

                # world_frame_object_pose_array.poses.append(
                #     camera_frame_pose
                # )
                self.get_logger().info(
                    "Camera frame pose: "
                    f"{camera_frame_pose.position.x}, "
                    f"{camera_frame_pose.position.y}, "
                    f"{camera_frame_pose.position.z}"
                )
                self.get_logger().info(
                    "World frame pose: "
                    f"{world_frame_object_pose.position.x}, "
                    f"{world_frame_object_pose.position.y}, "
                    f"{world_frame_object_pose.position.z}"
                )

        self.detection_position_transformed_pub.publish(world_frame_object_pose_array)

    def detection_pointcloud_callback(self, msg):
        """Handle detection point cloud message"""
        self.get_logger().info(
            f"Received {len(msg.point_clouds)} point clouds from topic "
            f"{self.get_parameter('detection_point_clouds_topic').value}"
        )

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = self.map_frame_id

        transformed_point_clouds = PointCloudArray(header=header)

        for point_cloud in msg.point_clouds:
            transformed_points = []
            for point in pointcloud2_to_xyz_array(point_cloud):
                og_point = PointStamped()
                og_point.header = point_cloud.header
                og_point.point.x = point[0]
                og_point.point.y = point[1]
                og_point.point.z = point[2]
                transformed_point = self.transform_point(
                    og_point,
                    from_frame=self.camera_frame_id,
                    to_frame=self.map_frame_id,
                    stamp=rclpy.time.Time(),
                )
                if transformed_point is not None:
                    transformed_points.append(
                        [
                            transformed_point.point.x,
                            transformed_point.point.y,
                            transformed_point.point.z,
                        ]
                    )
            self.get_logger().info(f"Pointcloud length: {len(transformed_points)}")
            point_cloud_msg = array_to_pointcloud2(
                np.array(transformed_points, dtype=np.float32),
                frame_id=self.map_frame_id,
                stamp=msg.header.stamp,
            )
            # self.detection_pointcloud_transformed_pub.publish(point_cloud_msg)
            point_cloud.header = header
            self.detection_pointcloud_transformed_pub.publish(point_cloud)

            transformed_point_clouds.point_clouds.append(point_cloud_msg)

        # self.detection_pointcloud_transformed_pub.publish(transformed_point_clouds)

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

            # self.get_logger().info(
            #     f"Transformation: {trans.transform.translation}\n {trans.transform.rotation}"
            # )

            # Transform the point
            transformed = do_transform_point(ps, trans)
            return transformed

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF transform failed: {e}")
            return None

    def is_object_new(self, object_class, object_position):
        for cur_obs in self.current_objects:
            if (
                cur_obs.class_id == object_class
                and np.linalg.norm(cur_obs.position - object_position)
                < self.new_object_threshold
            ):
                return False
        return True


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
