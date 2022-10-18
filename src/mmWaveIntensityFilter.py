#!/usr/bin/env python3
import numpy as np
import rospy
from std_msgs.msg import String
from ti_mmwave_rospkg.msg import RadarScan
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import ros_numpy
import visualization_msgs
from visualization_msgs.msg import Marker


# ****** NOTE ******
#apt install ros-noetic-ros-numpy

MAX_DEPTH = 3.0 # meters

pub = rospy.Publisher('/ti_mmwave/max_intensity_marker', Marker, queue_size=10)

# Create a marker and set some defaults
marker = Marker()
marker.header.frame_id = "ti_mmwave_1"
marker.id = 0;
marker.type = Marker.CUBE
marker.action = Marker.ADD
marker.pose.orientation.w = 1.0; # For unit quaternion
marker_scale = 0.25
marker.scale.x = marker_scale;
marker.scale.y = marker_scale;
marker.scale.z = marker_scale;
marker.color.a = 1.0;
marker.color.r = 0.0;
marker.color.g = 1.0;
marker.color.b = 0.0;


def callback(data):
	pc = ros_numpy.numpify(data)

	# Filter out all points farther than some distance
	# Using ROS coordinate conventions, X is forward and back
	good_indices = pc['x'] <= MAX_DEPTH
	if np.count_nonzero(good_indices) == 0:
		return
	pc = pc[good_indices]

	# Find the point with the max intensity
	max_intens_ind = np.argmax(pc['intensity'])

	# Set the marker data to publish
	# XYZ of the marker are the only fields that change
	marker.header.stamp = rospy.get_rostime()
	marker.pose.position.x = pc['x'][max_intens_ind];
	marker.pose.position.y = pc['y'][max_intens_ind];
	marker.pose.position.z = pc['z'][max_intens_ind];
	#print(marker.pose.position)
	#print("[%f, %f, %f]" % (marker.pose.position.x, marker.pose.position.y, marker.pose.position.z))

	pub.publish(marker)


def listener():
	rospy.init_node('mmwave_intensity_filter', anonymous=True)

	#rospy.Subscriber("/ti_mmwave/radar_scan", RadarScan, callback)
	rospy.Subscriber("/ti_mmwave/radar_scan_pcl_1", PointCloud2, callback)

	# spin() simply keeps python from exiting until this node is stopped
	rospy.spin()


if __name__ == '__main__':
	listener()

