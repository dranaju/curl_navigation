#! /usr/bin/env python3
import time
import rospy
from std_msgs.msg import *
from sensor_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

bridge = CvBridge()

def image_callback(data):
    # print('aqui')
    try:
        cv_image = bridge.imgmsg_to_cv2(data, data.encoding)
        nan_location = np.isnan(cv_image)
        # print(aux1.max())
        cv_image[nan_location] = np.nanmax(cv_image)
        # print(cv_image.max())
        norm_image =  (cv_image)*255./5.
        norm_image[0,0] = 255.
        norm_image = norm_image.astype('uint8')
        print(norm_image.min(), norm_image.max())

        # norm_image = cv2.normalize(norm_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # print(norm_image.max())
        norm_image = cv2.resize(norm_image.copy(), (920, 480), interpolation = cv2.INTER_AREA)
        norm_image = cv2.cvtColor(norm_image, cv2.COLOR_GRAY2BGR)
        # cv_image = cv2.resize(cv_image.copy(), (200, 200), interpolation = cv2.INTER_AREA)
        #print('aqui12')
        # print(norm_image.shape)
        # cv2.imshow("Image window", cv_image)
        # cv2.waitKey(3)
        cv2.imshow("Image depth", norm_image)
        cv2.waitKey(3)

    except CvBridgeError as e:
        print(e)
        
def image_callback_rgb(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')
        # print(data.encoding)
        cv_image = cv2.resize(cv_image.copy(), (920, 480), interpolation = cv2.INTER_AREA)
        # print(cv_image.shape)
        cv2.imshow("Image window", cv_image)
        cv2.imwrite('image_sim.png', cv_image)
        cv2.waitKey(3)

    except CvBridgeError as e:
        print(e)
        
if __name__ == "__main__":
    
    rospy.init_node("test", anonymous=False)
    rospy.Subscriber("/hydrone_aerial_underwater/camera/depth/image_raw", Image, image_callback)
    # # rospy.sleep(2)
    # rospy.Subscriber("/hydrone_aerial_underwater/camera/rgb/image_raw", Image, image_callback_rgb)

    rospy.spin()