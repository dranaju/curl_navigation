
#! /usr/bin/env python
import time
import rospy
from std_msgs.msg import *
from std_srvs.srv import *
from sensor_msgs.msg import *
from geometry_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2

bridge = CvBridge()
aux = False

def image_callback(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        #print('aqui12')
        # print(cv_image.shape)
        cv_image = cv2.resize(cv_image, (640,480))
        # print(cv_image.shape)
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)
    except CvBridgeError as e:
        print(e)

def depth_map_callback(data):
    global aux
    try:
        cv_map = bridge.imgmsg_to_cv2(data, '32FC1')
        print(cv_map.shape)
        # cv_depth = np.zeros(cv_map.shape, dtype=np.uint8)
        # # print(cv_depth)
        # nan_location = np.isnan(cv_map)
        # # print(aux1.max())
        # cv_map[nan_location] = np.nanmax(cv_map)
        # # cv2.normalize(cv_map, cv_depth, 0, 255, cv2.NORM_MINMAX)
        # cv_depth = cv_map*255/cv_map.max()
        # # cv2.convertScaleAbs(cv_map, cv_depth)
        # cv_depth = cv_depth.astype('uint8')
        # if not aux:
        #     print(cv_depth)
        #     print(np.max(cv_depth))
        #     print(np.min(cv_depth))
        #     aux = True
        # cv_depth = cv2.resize(cv_depth, (640,480))
        # print(np.max(cv_depth))
        # print(np.min(cv_depth))
        # print(cv_depth.shape)
        # cv2.imshow('Image depth', cv_depth)
        # cv2.waitKey(3)
    except CvBridgeError as e:
        print(e)
        
        
if __name__ == "__main__":
    
    rospy.init_node("test", anonymous=False)
    rospy.Subscriber("/camera/rgb/image_raw", Image, image_callback)
    # rospy.Subscriber("/camera/rgb/image_raw/compressedDepth", CompressedImage, depth_map_callback)

    rospy.spin()
