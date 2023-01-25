#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #
# Author-Contribution: Dranaju #

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import cv2
from respawnGoal import Respawn
import copy
from collections import deque
import os


#---Directory Path---#
dirPath = os.path.dirname(os.path.realpath(__file__))


class Env():
    def __init__(self, action_dim=2):
        self.goal_x = 0
        self.goal_y = 0
        self.goal_z = 0
        self.heading = 0
        self.heading_z = 0
        self.initGoal = True
        self.get_goalbox = False
        self.heading = 0
        self.position = Pose()
        self.pub_cmd_vel = self.pub_cmd_vel = rospy.Publisher('/hydrone_aerial_underwater/cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('/hydrone_aerial_underwater/ground_truth/odometry', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_world', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
        self.past_distance = 0.
        self.reward_distance = 0.
        self.stopped = 0
        self.action_dim = action_dim
        self.bridge = CvBridge()
        self._frames = deque([], maxlen=3)
        self._frames_print = deque([], maxlen=3)
        self.image_sub = rospy.Subscriber('/hydrone_aerial_underwater/camera/rgb/image_raw', Image, self.image_callback)
        # self.image_sub = rospy.Subscriber("/hydrone_aerial_underwater/camera/depth/image_raw", Image, self.image_callback)
        rospy.sleep(1)
        #Keys CTRL + c will stop script
        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        #you can stop turtlebot by publishing an empty Twist
        rospy.loginfo("Stopping TurtleBot")
        self.pub_cmd_vel.publish(Twist())
        rospy.sleep(1)

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            self.cv_image = cv2.resize(cv_image.copy(), (100, 100), interpolation = cv2.INTER_AREA)
            # cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # nan_location = np.isnan(cv_image)
            # print(aux1.max())
            # cv_image[nan_location] = np.nanmax(cv_image)
            # norm_image =  (cv_image)*255./5.
            # norm_image[0,0] = 255.
            # norm_image = norm_image.astype('uint8')

            # norm_image = cv2.normalize(norm_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # norm_image = cv2.resize(norm_image.copy(), (100, 100), interpolation = cv2.INTER_AREA)
            # self.cv_image = cv2.cvtColor(norm_image, cv2.COLOR_GRAY2BGR)
            cv2.imshow("Current_Observation", cv2.resize(self.cv_image.copy(), (500, 400), interpolation = cv2.INTER_AREA))
            cv2.waitKey(3)
        except CvBridgeError as e:
            print(e)

    def getGoalDistace(self):
        goal_distance = math.sqrt((self.goal_x - self.position.x)**2 + (self.goal_y - self.position.y)**2 + (self.goal_z - self.position.z)**2)
        self.past_distance = goal_distance

        return goal_distance

    def getOdometry(self, odom):
        self.past_position = copy.deepcopy(self.position)
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)
        self.heading_z = math.atan2(self.goal_z - self.position.z, math.sqrt((self.goal_x - self.position.x)**2 + (self.goal_y - self.position.y)**2))
        # rospy.loginfo("%s", goal_angle_z)

        heading = goal_angle - yaw
        #print 'heading', heading
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 3)

    def getState(self, scan):
        scan_range = []
        heading = self.heading
        min_range = 0.625
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf') or scan.ranges[i] == float('inf'):
                scan_range.append(20)
            elif np.isnan(scan.ranges[i]) or scan.ranges[i] == float('nan'):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])


        if (min_range > min(scan_range) > 0) or self.position.z < -1.1 or self.position.z > 4.8:
            done = True

        current_distance = math.sqrt((self.goal_x - self.position.x)**2 + (self.goal_y - self.position.y)**2 + (self.goal_z - self.position.z)**2)

        self.reward_distance = self.past_distance - current_distance

        self.past_distance = copy.deepcopy(current_distance)

        # print('distance', current_distance)

        if current_distance < 0.4:
            self.get_goalbox = True

        position_agent = [self.position.x, self.position.y, self.position.z]
        positon_goal = [self.goal_x, self.goal_y, self.goal_z]

        pose_target = [self.heading, self.heading_z, current_distance] + position_agent + positon_goal
        # print('pose_target', pose_target)

        return self.cv_image.transpose(2, 0, 1).copy(), pose_target, done

        

    def setReward(self, state, done):
        reward = 0.1*self.reward_distance

        if (self.position.z > -0.23) and (self.position.z < -0.05):
            if reward < 0.:
                reward = reward - 0.01
            if reward > 0.: 
                reward = reward + 0.01

        if self.position.z < -0.75:
            reward = -0.1

        if done:
            rospy.loginfo("Collision!!")
            # reward = -500.
            reward = -1.
            self.pub_cmd_vel.publish(Twist())
            # self.reset()
            self.respawn_goal.counter = 0
            

        if self.get_goalbox:
            rospy.loginfo("Goal!! ")
            reward = 1.
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y, self.goal_z = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward, done

    def step(self, action):
        linear_vel = action[0] + 0.01
        ang_vel = action[1]
        z_vel = action[2]

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.linear.z = z_vel
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/hydrone_aerial_underwater/scan', LaserScan, timeout=5)
            except:
                pass

        state, pose_target, done = self.getState(data)
        reward, done = self.setReward(state, done)

        self._frames.append(state)

        self._frames_print.append(state.transpose(1, 2, 0).copy())

        # print('aqui', self._frames_print)

        cv2.imwrite(dirPath + '/observation.png', np.concatenate(list(self._frames_print), axis=0))

        return self._get_obs(), pose_target, reward, done

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/hydrone_aerial_underwater/scan', LaserScan, timeout=5)
            except:
                pass
        
        state, pose_target, _ = self.getState(data)
        # print('aqui2')
        for _ in range(3):
            # print('aqui')
            self._frames.append(state)

        for _ in range(3):
            self._frames_print.append(state.transpose(1, 2, 0).copy())

        if self.initGoal:
            self.goal_x, self.goal_y, self.goal_z = self.respawn_goal.getPosition()
            self.initGoal = False
        else:
            self.goal_x, self.goal_y, self.goal_z = self.respawn_goal.getPosition(True, delete=True, init=False)
        
        self.goal_distance = self.getGoalDistace()

        # print('aqui2')

        obs = self._get_obs()

        return obs, pose_target

    def _get_obs(self):
        assert len(self._frames) == 3
        return np.concatenate(list(self._frames), axis=0)
