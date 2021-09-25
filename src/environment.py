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
        self.heading = 0
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.get_odometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.past_distance = 0.
        self.stopped = 0
        self.action_dim = action_dim
        self.bridge = CvBridge()
        self._frames = deque([], maxlen=3)
        self._frames_print = deque([], maxlen=3)
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        #Keys CTRL + c will stop script
        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        #you can stop turtlebot by publishing an empty Twist
        rospy.loginfo("Stopping TurtleBot")
        self.pub_cmd_vel.publish(Twist())
        rospy.sleep(1)

    def image_callback(self, data):
        try:
            cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')
            self.cv_image = cv2.resize(cv_image.copy(), (100, 100), interpolation = cv2.INTER_AREA)
            cv2.imshow("Current_Observation", cv_image)
            cv2.waitKey(3)
        except CvBridgeError as e:
            print(e)

    def get_goal_distance(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        self.past_distance = goal_distance

        return goal_distance

    def get_odometry(self, odom):
        self.past_position = copy.deepcopy(self.position)
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        #print 'yaw', yaw
        #print 'gA', goal_angle

        heading = goal_angle - yaw
        #print 'heading', heading
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 3)

    def getState(self, scan):
        scan_range = []
        # heading = self.heading
        min_range = 4.
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf') or scan.ranges[i] == float('inf'):
                scan_range.append(81.)
            elif np.isnan(scan.ranges[i]) or scan.ranges[i] == float('nan'):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])


        if min_range > min(scan_range) > 0:
            done = True

        # for pa in past_action:
        #     scan_range.append(pa)

        # self.current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        # # current_distance = self.get_goal_distance()
        # if self.current_distance < 0.2:
        #     self.get_goalbox = True

        return self.cv_image.transpose(2, 0, 1).copy(), done

    def setReward(self, state, done):
        reward = 0
        # current_distance = self.current_distance
        # heading = state[-2]
        #print('cur:', current_distance, self.past_distance)


        # distance_rate = (self.past_distance - current_distance) 
        # if distance_rate > 0:
        #     # reward = 200.*distance_rate
        #     reward = 0.

        # if distance_rate == 0:
        #     reward = 0.

        # if distance_rate <= 0:
        #     # reward = -8.
        #     reward = 0.

        #angle_reward = math.pi - abs(heading)
        #print('d', 500*distance_rate)
        #reward = 500.*distance_rate #+ 3.*angle_reward
        # self.past_distance = current_distance

        # a, b, c, d = float('{0:.3f}'.format(self.position.x)), float('{0:.3f}'.format(self.past_position.x)), float('{0:.3f}'.format(self.position.y)), float('{0:.3f}'.format(self.past_position.y))
        # if a == b and c == d:
        #     # rospy.loginfo('\n<<<<<Stopped>>>>>\n')
        #     # print('\n' + str(a) + ' ' + str(b) + ' ' + str(c) + ' ' + str(d) + '\n')
        #     self.stopped += 1
        #     if self.stopped == 20:
        #         rospy.loginfo('Robot is in the same 20 times in a row')
        #         self.stopped = 0
        #         done = True
        # else:
        #     # rospy.loginfo('\n>>>>> not stopped>>>>>\n')
        #     self.stopped = 0

        if done:
            rospy.loginfo("Collision!!")
            # reward = -500.
            reward = -10.
            self.pub_cmd_vel.publish(Twist())
            self.reset()

        # if self.get_goalbox:
        #     rospy.loginfo("Goal!!")
        #     # reward = 500.
        #     reward = 100.
        #     self.pub_cmd_vel.publish(Twist())
        #     if world:
        #         self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True, running=True)
        #         if target_not_movable:
        #             self.reset()
        #     else:
        #         self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
        #     self.goal_distance = self.get_goal_distance()
        #     self.get_goalbox = False

        return reward, done

    def step(self, action):
        linear_vel = action[0]
        ang_vel = action[1]

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data)
        reward, done = self.setReward(state, done)

        self._frames.append(state)

        self._frames_print.append(state.transpose(1, 2, 0).copy())

        cv2.imwrite(dirPath + '/aqui4.png', np.concatenate(list(self._frames_print), axis=0))

        return self._get_obs(), reward, done

    def reset(self):
        # print('aqui2_____________---')
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        print('aqui3_____________---')

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        # if self.initGoal:
        #     self.goal_x, self.goal_y = self.respawn_goal.getPosition()
        #     self.initGoal = False
        # else:
        #     self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)

        # self.goal_distance = self.get_goal_distance()
        state, _ = self.getState(data)
        for _ in range(3):
            self._frames.append(state)

        for _ in range(3):
            self._frames_print.append(state.transpose(1, 2, 0).copy())

        print('aqui4_____________---')

        return self._get_obs()

    def _get_obs(self):
        assert len(self._frames) == 3
        return np.concatenate(list(self._frames), axis=0)
