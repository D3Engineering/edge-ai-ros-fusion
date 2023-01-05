#!/usr/bin/env python3

'''
Camera + Radar fusion node - overlays the point-cloud data
from the radar on top of the image stream from the camera.
'''

import rospy
import sys
import cv2
import numpy as np
import ros_numpy # apt install ros-noetic-ros-numpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2


MAX_DEPTH = 3.0 # meters
PCL_PERSIST_FRAME_CNT = 2 # image frames


def draw_radar_points(img_rect, K, radar_data):
	# Extract x,y,z points from radar data
	pts_3d = np.zeros((len(radar_data), 4), dtype=np.float32)
	# Note the conversion here from ROS to OpenCV coordinates
	pts_3d[:,0] = -radar_data['y']
	pts_3d[:,1] = -radar_data['z']
	pts_3d[:,2] =  radar_data['x']
	pts_3d[:,3] =  1.0

	# Transform points from the radar to the camera's coordinate system
	# Hard coding a y translation of +28mm to test
	radar_to_camera_tf = np.eye(4)
	if 1:
		# Disable this to visualize without correction
		radar_to_camera_tf[1,3] = 0.028
	# Note the transpose to convert from row vector to column vector for numpy broadcasting
	pts_3d = radar_to_camera_tf @ pts_3d.T

	# Project points onto rectified image
	# Homogenize K
	K_homog = np.hstack((K, np.zeros((3,1))))
	pts_2d_homog = K_homog @ pts_3d

	pts_2d = np.zeros((2,len(radar_data)), dtype=np.float32)
	pts_2d[0,:] = pts_2d_homog[0,:] / pts_2d_homog[2,:]
	pts_2d[1,:] = pts_2d_homog[1,:] / pts_2d_homog[2,:]

	# Create colormap values for encoding depth with a colormap
	depth_data = radar_data['x'] # Note that this is actually Z (depth)
	# Clip invalid negative depths to 0 (just for normalization, the projection check will likely fail on these)
	depth_data[depth_data < 0] = 0
	# Normalize 0-MAX_DEPTH meters to 0-255 for the colormap
	depth_data = depth_data*255/MAX_DEPTH
	depth_colors = cv2.applyColorMap(depth_data.astype(np.uint8), cv2.COLORMAP_RAINBOW).squeeze().astype(np.uint8)

	# TODO - scale point sizes based on intensity
	# Need to add intensity to the PCL first
	#intensity = radar_data['intensity']

	# Transpose to back to list of points, and cast to int for pixel indexing
	for pt, color in zip(pts_2d.T.astype(np.int32), depth_colors):
		# Bounds check the projection to ensure it lands on the imager
		if (pt[0] > 0 and pt[0] < camera_info.width) and (pt[1] > 0 and pt[1] < camera_info.height):
			#cv2.drawMarker(img_rect, tuple(pt), color=[0,255,0], markerType=cv2.MARKER_CIRCLE, thickness=5, markerSize=40)
			cv2.circle(img_rect, tuple(pt), radius=5, color=color.tolist(), thickness=-1, lineType=cv2.FILLED)


def processImage(msg):
	try:
		# convert sensor_msgs/Image to OpenCV Image
		bridge = CvBridge()
		img = bridge.imgmsg_to_cv2(msg, "bgr8")

		if len(latest_radar_data):
			for radar_data_frame in latest_radar_data:
				draw_radar_points(img, K, radar_data_frame)

		imgMsg = bridge.cv2_to_imgmsg(img, "bgr8")
		imagePub.publish(imgMsg)

	except Exception as err:
		print(err)


def pcl_callback(data):
	pc = ros_numpy.numpify(data)

	# Filter out all points farther than some distance
	# Using ROS coordinate conventions, X is forward and back
	good_indices = pc['x'] <= MAX_DEPTH
	if np.count_nonzero(good_indices) == 0:
		return
	pc = pc[good_indices]

	#print(pc)
	global latest_radar_data
	# This is a hacky way to add persistance to the points
	latest_radar_data.append(pc)
	if len(latest_radar_data) > PCL_PERSIST_FRAME_CNT:
		latest_radar_data.pop(0)


def start_node():

	print("Python version: %s" % str(sys.version))
	print("OpenCV version: %s" % str(cv2.__version__))

	rospy.init_node('fusion_node', anonymous=True)
	rospy.loginfo('fusion_node started')

	# Get camera name from parameter server
	global camera_name
	camera_name = rospy.get_param("~camera_name", "camera")
	camera_info_topic = "{}/camera_info".format(camera_name)
	rospy.loginfo("Waiting on camera_info: %s" % rospy.resolve_name(camera_info_topic))

	# Wait until we have valid calibration data before starting
	global camera_info
	camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)
	rospy.loginfo("Camera intrinsic matrix: %s" % str(camera_info.K))
	rospy.loginfo("Camera distortion coefficients: %s" % str(camera_info.D))

	# Pre-compute distortion correction map for efficiency
	global map_x
	global map_y
	global K
	DIM = (camera_info.width, camera_info.height)
	K = np.array(camera_info.K).reshape((3,3)) # Un-flatten
	D = np.array(camera_info.D)
	assert len(D) == 4, "Error, expecting len(D) == 4"
	image_size = (camera_info.width, camera_info.height)

	# Setup subscriber for the raw image stream
	rospy.loginfo("Subscribing to: %s" % rospy.resolve_name("/image_raw"))
	rospy.Subscriber("/image_raw", Image, processImage)


	# Setup subscriber for the radar data
	radar_name = rospy.get_param("~radar_name", "/ti_mmwave")
	radar_topic = "{}/radar_scan_pcl".format(radar_name)
	rospy.loginfo("Subscribing to radar: %s" % rospy.resolve_name(radar_topic))
	rospy.Subscriber(radar_topic, PointCloud2, pcl_callback)

	global latest_radar_data
	latest_radar_data = []

	# Setup publisher for undistorted image stream
	global imagePub
	camera_img_topic = "{}/image_fused".format(camera_name)
	imagePub = rospy.Publisher(camera_img_topic, Image, queue_size=1)

	rospy.spin()


if __name__ == '__main__':
	try:
		start_node()
	except rospy.ROSInterruptException:
		pass
