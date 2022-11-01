#!/usr/bin/env python3

'''
TODO
'''

import rospy
import sys
import cv2
import numpy as np
#import ros_numpy # apt install ros-noetic-ros-numpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import PointCloud2
from ti_mmwave_rospkg.msg import RadarTrackArray
from common_msgs.msg import Detection2D
import sensor_msgs.point_cloud2 as pc2
#from icecream import ic # python3 -m pip install icecream

# Thresholds for visualization
MAX_DEPTH = 4.0		# meters
MIN_VELOCITY = 0.1	# m/s
TRACK_PERSIST_TIME = rospy.Duration.from_sec(0.25) # Seconds

# See ti_viz_nodes/src/viz_objdet.h for full list of supported classes
valid_objs = {}
valid_objs[1] = 'person'
valid_objs[3] = 'car'
valid_objs[8] = 'truck'
#valid_objs[33] = 'suitcase'


def draw_text_block(img, text_bot_left, header, data):
	font = cv2.FONT_HERSHEY_DUPLEX
	scale = 0.5
	thickness = 1
	color = (255,255,255)
	buffer_pix = 10 # Buffer pixels in the y (row) dimension
	text_size, _ = cv2.getTextSize(header, font, scale, thickness)
	text_bot_left = (text_bot_left[0], text_bot_left[1]+text_size[1]+buffer_pix)
	cv2.putText(img, header, text_bot_left, font, scale, color, thickness, cv2.LINE_AA)
	data_text = ' x: {:3.2f}'.format(data[0])
	text_size, _ = cv2.getTextSize(data_text, font, scale, thickness)
	text_bot_left = (text_bot_left[0], text_bot_left[1]+text_size[1]+buffer_pix)
	cv2.putText(img, data_text, text_bot_left, font, scale, color, thickness, cv2.LINE_AA)
	data_text = ' y: {:3.2f}'.format(data[1])
	text_size, _ = cv2.getTextSize(data_text, font, scale, thickness)
	text_bot_left = (text_bot_left[0], text_bot_left[1]+text_size[1]+buffer_pix)
	cv2.putText(img, data_text, text_bot_left, font, scale, color, thickness, cv2.LINE_AA)
	data_text = ' z: {:3.2f}'.format(data[2])
	text_size, _ = cv2.getTextSize(data_text, font, scale, thickness)
	text_bot_left = (text_bot_left[0], text_bot_left[1]+text_size[1]+buffer_pix)
	cv2.putText(img, data_text, text_bot_left, font, scale, color, thickness, cv2.LINE_AA)
	return text_bot_left


def draw_result(img, K, cnn_data, radar_data):

	# Keep track of the 2D projections of the radar tracker so we can associate them with CNN objects
	track_projections = {}

	# radar_data is a dict of track_id->track_data for all track info received between the previous image frame and now.
	for track_id in radar_data:
		track_data = radar_data[track_id][0]

		# Note the conversion here from ROS to OpenCV coordinates
		pt_3d = np.array([-track_data.posy, -track_data.posz, track_data.posx, 1.0], dtype=np.float32)

		# Transform points from the radar to the camera's coordinate system
		# Hard coding a y translation of +28mm to test
		radar_to_camera_tf = np.eye(4)
		radar_to_camera_tf[1,3] = -0.028 # Disable this to visualize without correction

		# Note the transpose to convert from row vector to column vector for numpy broadcasting
		pt_3d_tf = radar_to_camera_tf @ pt_3d.T

		# Project points onto rectified image
		# Homogenize K
		K_homog = np.hstack((K, np.zeros((3,1))))
		pt_2d_homog = K_homog @ pt_3d_tf

		# Dehomogenize and c0ast to int for pixel indexing
		pt_2d = (int(pt_2d_homog[0]/pt_2d_homog[2]), int(pt_2d_homog[1]/pt_2d_homog[2]))

		# Bounds check the projection to ensure it lands on the imager
		if (pt_2d[0] > 0 and pt_2d[0] < camera_info.width) and (pt_2d[1] > 0 and pt_2d[1] < camera_info.height):
			track_projections[track_id] = pt_2d

			cv2.drawMarker(img, pt_2d, color=[0,255,0], markerType=cv2.MARKER_SQUARE, thickness=3, markerSize=30)

			# Draw velocity vector as a 3D arrow projected into 2D
			# We have the origin point of the arrow from the position of the tracked object
			# Now determine the 2D location of the point of the arrow
			# Note: velocity is in m/s, and the length of the 3D arrow reflects this.
			vel_3d = np.array([-track_data.vely, -track_data.velz, track_data.velx, 1.0], dtype=np.float32)

			# Filter out velocities < MIN_VELOCITY
			if np.linalg.norm(vel_3d[:3]) < MIN_VELOCITY: # m/s
				return

			# Create 3D vector based on tracked pose + velocity vector
			vel_3d[:3] += pt_3d.flatten()[:3]
			vel_3d_tf = radar_to_camera_tf @ vel_3d.T

			# Project, dehomogenize, and draw
			vel_2d_homog = K_homog @ vel_3d_tf.T
			vel_2d = (int(vel_2d_homog[0]/vel_2d_homog[2]), int(vel_2d_homog[1]/vel_2d_homog[2]))
			cv2.arrowedLine(img, pt_2d, vel_2d, color=[0,0,255], thickness=2)

	for bbox in cnn_data.bounding_boxes:
		# Filter based on classes in valid_objs
		if bbox.label_id in valid_objs.keys():

			top_left = (bbox.xmin, bbox.ymin)
			bottom_right = (bbox.xmax, bbox.ymax)

			# Draw bounding box
			cv2.rectangle(img, top_left, bottom_right, color=[255,0,0], thickness=2)

			# Draw object class text
			scale = 0.5
			thickness = 1
			buffer_pix = 10 # Buffer pixels in the y (row) dimension
			obj_text = 'Object: {}'.format(valid_objs[bbox.label_id])
			text_size, _ = cv2.getTextSize(obj_text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
			text_bot_left = (top_left[0], top_left[1]+text_size[1])
			cv2.putText(img, obj_text, text_bot_left, cv2.FONT_HERSHEY_DUPLEX, fontScale=scale, \
						color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)

			# Draw radar tracker results
			if len(radar_data) == 0:
				continue

			# Search to see if we have a radar tracker result within this bbox
			for track_id in track_projections:
				# track_projections are the 2D projections of the radar tracker that were recorded earlier
				track_2d = track_projections[track_id]
				if track_2d[0] >= bbox.xmin and track_2d[0] <= bbox.xmax and track_2d[1] >= bbox.ymin and track_2d[1] <= bbox.ymax:
					# Got a correspondence, draw the track text on the bbox
					track_data = radar_data[track_id][0]
					text_bot_left = draw_text_block(img, text_bot_left, 'Position (m):', [-track_data.posy, -track_data.posz, track_data.posx])
					text_bot_left = draw_text_block(img, text_bot_left, 'Velocity (m/s):', [-track_data.vely, -track_data.velz, track_data.velx])
					text_bot_left = draw_text_block(img, text_bot_left, 'Acceleration (m/s^2):', [-track_data.accy, -track_data.accz, track_data.accx])
					break


def processImage(msg):
	global imagePub
	global latest_radar_data
	global latest_cnn_data

	# Copy radar data before processing to prevent "RuntimeError: dictionary changed size during iteration"
	radar_data = latest_radar_data.copy()

	try:
		# convert sensor_msgs/Image to OpenCV Image
		bridge = CvBridge()
		img = bridge.imgmsg_to_cv2(msg, "bgr8")

		enable_fusion = True
		if enable_fusion:
			#if radar_data is not None:
			#	draw_radar_points(img, K, radar_data)
			if latest_cnn_data is not None:
				draw_result(img, K, latest_cnn_data, radar_data)

		imgMsg = bridge.cv2_to_imgmsg(img, "bgr8")
		imagePub.publish(imgMsg)

		# Clear the array of tracker returns that we got between the last image and this one
		if 0:
			latest_radar_data = {}
		# Clear tracks that are older than the time threshold that we set
		# Note that because we're using a dict, each track_id will always have the most up to date data for it
		else:
			# Hack: Pop all old track_ids at once to prevent "RuntimeError: dictionary changed size during iteration"
			old_tracks = []
			for track_id in latest_radar_data:
				if rospy.Time.now() - latest_radar_data[track_id][1] > TRACK_PERSIST_TIME:
					old_tracks.append(track_id)
			[latest_radar_data.pop(key) for key in old_tracks]

	except Exception as err:
		print("Caught exception: %s" % str(err))


def radar_callback(data):
	global latest_radar_data

	#print("Num tracks: %i" % data.num_tracks)

	for track_idx in range(data.num_tracks):
		track = data.track[track_idx]

		# Filter out all points farther than some distance
		# Using ROS coordinate conventions, X is forward and back- a.k.a. depth
		if track.posx > MAX_DEPTH:
			return

		# Due to radar framerate being potentially higher than the camera, use a dict to store the latest tracker result for each track
		# This data will get cleared after receiving and processing an image
		latest_radar_data[track.tid] = (track, rospy.get_rostime())


def cnn_callback(data):
	global latest_cnn_data
	latest_cnn_data = data


def start_node():

	print("Python version: %s" % str(sys.version))
	print("OpenCV version: %s" % str(cv2.__version__))

	rospy.init_node('fusion_node', anonymous=True)
	rospy.loginfo('fusion_node started')

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

	# Pre-compute distortion correction map for efficiency
	global map_x
	global map_y
	global K
	DIM = (camera_info.width, camera_info.height)
	K = np.array(camera_info.K).reshape((3,3)) # Un-flatten
	D = np.array(camera_info.D)
	assert len(D) == 4, "Error, expecting len(D) == 4"
	image_size = (camera_info.width, camera_info.height)


	# Setup subscriber for the radar data
	rospy.Subscriber("/ti_mmwave/radar_trackarray", RadarTrackArray, radar_callback)
	global latest_radar_data
	# latest_radar_data is a dict and accumulates all of the tracked returns between image frames
	latest_radar_data = {}

	# Setup subscriber for CNN output
	cnn_output_topic = "/{}/vision_cnn/tensor".format(camera_name)
	rospy.Subscriber(cnn_output_topic, Detection2D, cnn_callback)
	global latest_cnn_data
	latest_cnn_data = None

	# Setup publisher for fused image stream
	global imagePub
	camera_img_topic = "/{}/image_fused".format(camera_name)
	imagePub = rospy.Publisher(camera_img_topic, Image, queue_size=1)

	# Setup subscriber for the raw image stream
	rospy.Subscriber("/image_raw", Image, processImage)


	rospy.spin()


if __name__ == '__main__':
	try:
		start_node()
	except rospy.ROSInterruptException:
		pass
