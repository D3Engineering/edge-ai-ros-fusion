#!/usr/bin/env python3

'''
Simple image undistorter node for fisheye distortion correction
Necessary b/c image_proc package doesn't support fisheye distortion
'''

# Python imports
import rospy
import sys
import cv2
import numpy as np
# Ros imports
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo


def processImage(msg):
	try:
		# convert sensor_msgs/Image to OpenCV Image
		bridge = CvBridge()
		img_raw = bridge.imgmsg_to_cv2(msg, "bgr8")

		# HACK
		# Sensor raw images are actually 1936x1096, but TDA instead uses 1936x1100
		# TODO - determine how to adjust here for this
		# For now, crop the first and last 2 rows off
		#img_raw = img_raw[2:-2,:]
		# Or last 4 rows are better?
		img_raw = img_raw[:-4,:]

		# Rectify image
		assert_dim_str = "Image to undistort needs to have same aspect ratio as the ones used in calibration"
		assert camera_info.width/camera_info.height == img_raw.shape[1]/img_raw.shape[0], assert_dim_str

		img_rect = cv2.remap(img_raw, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

		imgMsg = bridge.cv2_to_imgmsg(img_rect, "bgr8")
		imagePub.publish(imgMsg)

	except Exception as err:
		print(err)


def start_node():

	print("Python version: %s" % str(sys.version))
	print("OpenCV version: %s" % str(cv2.__version__))

	rospy.init_node('undistort_node', anonymous=True)
	rospy.loginfo('undistort_node started')

	# Get camera name from parameter server
	global camera_name
	camera_name = rospy.get_param("~camera_name", "camera")
	camera_info_topic = "/{}/camera_info".format(camera_name)
	rospy.loginfo("Waiting on camera_info: %s" % camera_info_topic)

	# Wait until we have valid calibration data before starting
	global camera_info
	camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)
	rospy.loginfo("Camera intrinsic matrix: %s" % str(camera_info.K))
	rospy.loginfo("Camera distortion coefficients: %s" % str(camera_info.D))

	# Pre-compute distortion correction map
	global map_x
	global map_y
	image_size = (camera_info.width, camera_info.height)
	K = np.array(camera_info.K).reshape((3,3)) # Un-flatten
	D = np.array(camera_info.D)

	if len(D) == 4:
		new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, image_size, R=np.eye(3), balance=1.0)
		map_x, map_y = cv2.fisheye.initUndistortRectifyMap(K, D, R=np.eye(3), P=new_K, size=image_size, m1type=cv2.CV_32FC1)

	else:
		#map_x, map_y   = cv2.initUndistortRectifyMap(K, D, np.eye(3), K, image_size, cv2.CV_32FC1)
		# alpha=0.0 crops to only sensible pixels
		new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, image_size, alpha=0.0, newImgSize=new_image_size)
		map_x, map_y = cv2.initUndistortRectifyMap(K, D, None, new_K, new_image_size, cv2.CV_32FC1)

	# Setup subscriber for the raw image stream
	rospy.Subscriber("image_raw", Image, processImage)

	# Setup publisher for undistorted image stream
	global imagePub
	camera_img_topic = "/{}/image_undistorted_rgb".format(camera_name)
	imagePub = rospy.Publisher(camera_img_topic, Image, queue_size=1)

	rospy.spin()


if __name__ == '__main__':
	try:
		start_node()
	except rospy.ROSInterruptException:
		pass
