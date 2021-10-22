#!/usr/bin/env python
# Authors: Junior Costa de Jesus #

import rospy
import numpy as np
import random
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from environment import Env
import gc
import math
import copy
from std_msgs.msg import Float32
import cv2


#---Directory Path---#
dirPath = os.path.dirname(os.path.realpath(__file__))


def action_unnormalized(action, high, low):
    action = low + (action + 1.0) * 0.5 * (high - low)
    action = np.clip(action, low, high)
    return action

action_dim = 2
state_dim  = 14
hidden_dim = 500
ACTION_V_MIN = 0.0 # m/s
ACTION_W_MIN = -2. # rad/s
ACTION_V_MAX = 0.22 # m/s
ACTION_W_MAX = 2. # rad/s

def evaluate():
    print('evaluated')

if __name__ == '__main__':
    rospy.init_node('curl_node')
    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    result = Float32()
    env = Env()

    action_shape = action_dim

    obs_shape = (3*3, 84, 84)
    pre_aug_obs_shape = (3*3, 100, 100)

    episode, episode_reward, done = 0, 0, True
    max_steps = 1000000
    initial_step = 0
    save_model_replay = True

    for step in range(initial_step, max_steps):
        
        if step % 1000 == 0:
            evaluate()
            if save_model_replay:
                print('saved model and replay memory')
        
        if done:
            
            obs = env.reset()
            done = False
            episode_reward = 0
            episode += 1

            print("*********************************")
            print('Episode: ' + str(episode) + ' training')
            print('Step: ' + str(step) + ' training')
            print('Reward average per ep: ' + str(episode_reward))
            print("*********************************")

        if step < 100: #1000
            action = np.array([
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1)
                ]) 
            unnorm_action = np.array([
                action_unnormalized(action[0], ACTION_V_MAX, 0), 
                action_unnormalized(action[1], ACTION_W_MAX, ACTION_W_MIN)
                ])
        else:
            action = np.array([
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1)
                ]) 
            unnorm_action = np.array([
                action_unnormalized(action[0], ACTION_V_MAX, 0), 
                action_unnormalized(action[1], ACTION_W_MAX, ACTION_W_MIN)
                ])

        if step >= 100: #1000
            num_updates = 1 
            for _ in range(num_updates):
                print('update')

        print('action ', unnorm_action)

        next_obs, reward, done = env.step(unnorm_action)
        

        print('reward ', reward)
        # print('state ', obs.shape)

        episode_reward += reward
        #replay_buffer.add(obs, action, reward, next_obs, done)
        if reward < -1.:
            print('\n----collide-----\n')
            for i in range(3):
                print('aqui2')
                # replay_buffer.add(obs, action, reward, next_obs, done)

        obs = next_obs
        # episode_step += 1