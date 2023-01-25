#!/usr/bin/env python
# Authors: Junior Costa de Jesus #

import rospy
import os
import json
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32
from environment_hydrone_image_depth import Env
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gc
import torch.nn as nn
import math
import copy
import cv2
from torch.utils.tensorboard import SummaryWriter

#---Directory Path---#
dirPath = os.path.dirname(os.path.realpath(__file__))

#****************************************************


import torch
import numpy as np
import torch.nn as nn
# import gym
import os
from collections import deque
import random
from torch.utils.data import Dataset, DataLoader
from skimage.util.shape import view_as_windows

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""
    def __init__(
                self, obs_shape, action_shape, 
                capacity, batch_size, device,
                image_size=84,transform=None,
                alpha=0.6,beta_start = 0.4,beta_frames=100000
                ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        # print('aqui1', obs_shape)
        # print('aqui1', *obs_shape)
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        self.obses = dict()
        self.obses['anchor'] = np.empty((capacity, obs_shape[0], obs_shape[1], obs_shape[2]), dtype=obs_dtype)
        self.obses['pos'] = np.empty((capacity, obs_shape[0], obs_shape[1], obs_shape[2]), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, obs_shape[0], obs_shape[1], obs_shape[2]), dtype=obs_dtype)
        self.actions = np.empty((capacity, action_shape), dtype=np.float32)
        self.poses = np.empty((capacity, 3+6), dtype=np.float32)
        self.next_poses = np.empty((capacity, 3+6), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1 #for beta calculation
        self.priorities = np.zeros((capacity,), dtype=np.float32)


    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.
        
        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent 
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)


    

    def add(self, obs, obs_depth, pose, action, reward, next_obs, next_pose, done):

        max_prio = self.priorities.max() if (self.idx == 0 and not self.full) else 1.0
        # print('max_prio', max_prio)

        np.copyto(self.obses['anchor'][self.idx], obs)
        np.copyto(self.obses['pos'][self.idx], obs_depth)
        np.copyto(self.poses[self.idx], pose)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.next_poses[self.idx], next_pose)
        np.copyto(self.not_dones[self.idx], not done)

        self.priorities[self.idx] = max_prio
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_proprio(self):
        
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
        
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        poses = torch.as_tensor(self.poses[idxs], device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        next_poses = torch.as_tensor(self.next_poses[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, poses, actions, rewards, next_obses, next_poses, not_dones

    def sample_cpc(self):
        if not self.full:
            N = self.idx
        else:
            N = self.capacity

        if N == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.idx]

        # calc P = p^a/sum(p^a)
        probs  = prios ** self.alpha
        P = probs/probs.sum()
        
        #gets the indices depending on the probability p
        indices = np.random.choice(N, self.batch_size, p=P)
        
        beta = self.beta_by_frame(self.frame)
        self.frame+=1
                
        #Compute importance-sampling weight
        weights  = (N * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max() 
        weights  = np.array(weights, dtype=np.float32) 
        

        # start = time.time()
        # idxs = np.random.randint(
        #     0, self.capacity if self.full else self.idx, size=self.batch_size
        # )
        # print(type(self.obses))
        obses = np.array([self.obses['anchor'][idxs] for idxs in indices])
        next_obses = np.array([self.next_obses[idxs] for idxs in indices])
        # pos = obses.copy()
        pos = np.array([self.obses['pos'][idxs] for idxs in indices])

        # print('------------------')
        # rgb = np.dstack((obses[0][0], obses[0][1], obses[0][2]))
        # print(rgb.shape)

        # cv2.imwrite(dirPath + '/observation_rgb.png', rgb)

        # rgb = np.dstack((pos[0][0], pos[0][1], pos[0][2]))
        # print(rgb.shape)

        # cv2.imwrite(dirPath + '/observation_rgb_depth.png', rgb)

        # obses = random_crop(obses, self.image_size)
        # next_obses = random_crop(next_obses, self.image_size)
        # pos = random_crop(pos, self.image_size)
    
        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        poses = torch.as_tensor(np.array([self.poses[idxs] for idxs in indices]), device=self.device) 
        next_poses = torch.as_tensor(np.array([self.next_poses[idxs] for idxs in indices]), device=self.device)

        actions = torch.as_tensor(np.array([self.actions[idxs] for idxs in indices]), device=self.device) 
        rewards = torch.as_tensor(np.array([self.rewards[idxs] for idxs in indices]), device=self.device)
        not_dones = torch.as_tensor(np.array([self.not_dones[idxs] for idxs in indices]), device=self.device)

        pos = torch.as_tensor(pos, device=self.device).float()
        cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos,
                          time_anchor=None, time_pos=None)

        return obses, poses, actions, rewards, next_obses, next_poses, not_dones, cpc_kwargs, indices, weights

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end
            self.last_save = end
        print('Load memory until:' + str(self.idx))

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio) 

    # def __getitem__(self, idx):
    #     idx = np.random.randint(
    #         0, self.capacity if self.full else self.idx, size=1
    #     )
    #     idx = idx[0]
    #     obs = self.obses[idx]
    #     action = self.actions[idx]
    #     reward = self.rewards[idx]
    #     next_obs = self.next_obses[idx]
    #     not_done = self.not_dones[idx]

    #     if self.transform:
    #         obs = self.transform(obs)
    #         next_obs = self.transform(next_obs)

    #     return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity 

# class FrameStack(gym.Wrapper):
#     def __init__(self, env, k):
#         gym.Wrapper.__init__(self, env)
#         self._k = k
#         self._frames = deque([], maxlen=k)
#         shp = env.observation_space.shape
#         self.observation_space = gym.spaces.Box(
#             low=0,
#             high=1,
#             shape=((shp[0] * k,) + shp[1:]),
#             dtype=env.observation_space.dtype
#         )
#         self._max_episode_steps = env._max_episode_steps

#     def reset(self):
#         obs = self.env.reset()
#         for _ in range(self._k):
#             self._frames.append(obs)
#         return self._get_obs()

#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)
#         self._frames.append(obs)
#         return self._get_obs(), reward, done, info

#     def _get_obs(self):
#         assert len(self._frames) == self._k
#         return np.concatenate(list(self._frames), axis=0)


def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0,:,:, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs

def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 43, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32,output_logits=False):
        super(PixelEncoder, self).__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM_64[num_layers] if obs_shape[-1] == 64 else OUT_DIM[num_layers] 
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        # print('-------')
        # print(conv.size())

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    # def log(self, L, step, log_freq):
    #     if step % log_freq != 0:
    #         return

    #     for k, v in self.outputs.items():
    #         L.log_histogram('train_encoder/%s_hist' % k, v, step)
    #         if len(v.shape) > 2:
    #             L.log_image('train_encoder/%s_img' % k, v[0], step)

    #     for i in range(self.num_layers):
    #         L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
    #     L.log_param('train_encoder/fc', self.fc, step)
    #     L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters,*args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    # def log(self, L, step, log_freq):
    #     pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits
    )

LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters
    ):
        super(Actor, self).__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim + 3+6, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape)
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, pose, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)

        # print('encoder_q----------------------------')
        # # print(obs)
        # print(type(obs))
        # print(obs.size())
        # print(pose)
        # print(type(pose))
        # print(pose.size())
        obs_pose = torch.cat([obs, pose], dim=1)
        # print(type(obs_pose))
        # print('-------------------------------------')

        mu, log_std = self.trunk(obs_pose).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    # def log(self, L, step, log_freq=LOG_FREQ):
    #     if step % log_freq != 0:
    #         return

    #     for k, v in self.outputs.items():
    #         L.log_histogram('train_actor/%s_hist' % k, v, step)

    #     L.log_param('train_actor/fc1', self.trunk[0], step)
    #     L.log_param('train_actor/fc2', self.trunk[2], step)
    #     L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super(QFunction, self).__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + 3+6 + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, pose, action):
        assert obs.size(0) == action.size(0)

        obs_pose = torch.cat([obs, pose], dim=1)
        obs_action = torch.cat([obs_pose, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters
    ):
        super(Critic, self).__init__()


        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape, hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape, hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, pose, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)

        # print('aqui13', obs.shape)

        q1 = self.Q1(obs, pose, action)
        q2 = self.Q2(obs, pose, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    # def log(self, L, step, log_freq=LOG_FREQ):
    #     if step % log_freq != 0:
    #         return

    #     self.encoder.log(L, step, log_freq)

    #     for k, v in self.outputs.items():
    #         L.log_histogram('train_critic/%s_hist' % k, v, step)

    #     for i in range(3):
    #         L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
    #         L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class CURL(nn.Module):
    """
    CURL
    """

    def __init__(self, obs_shape, z_dim, batch_size, critic, critic_target, output_type="continuous"):
        super(CURL, self).__init__()
        self.batch_size = batch_size

        self.encoder = critic.encoder

        self.encoder_target = critic_target.encoder 

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def encode(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

class CurlSacAgent(object):
    """CURL representation learning with SAC."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        num_layers=4,
        num_filters=32,
        cpc_update_freq=1,
        log_interval=100,
        detach_encoder=False,
        curl_latent_dim=128
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.curl_latent_dim = curl_latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)
        
        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        if self.encoder_type == 'pixel':
            # create CURL encoder (the 128 batch size is probably unnecessary)
            self.CURL = CURL(obs_shape, encoder_feature_dim,
                        self.curl_latent_dim, self.critic,self.critic_target, output_type='continuous').to(self.device)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            self.cpc_optimizer = torch.optim.Adam(
                self.CURL.parameters(), lr=encoder_lr
            )
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_type == 'pixel':
            self.CURL.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs, pose):
        # if obs.shape[-1] != self.image_size:
            # obs = center_crop_image(obs, self.image_size)

        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            pose = torch.FloatTensor([pose]).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(obs, pose, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs, pose):
        # if obs.shape[-1] != self.image_size:
            # obs = center_crop_image(obs, self.image_size)
 
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            pose = torch.FloatTensor([pose]).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, pose, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, pose, action, reward, next_obs, next_pose, not_done, step, weights, writer):
        weights    = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs, next_pose)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_pose, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs, pose, action, detach_encoder=self.detach_encoder)
        # critic_loss = F.mse_loss(current_Q1,
        #                          target_Q) + F.mse_loss(current_Q2, target_Q)
        td_error1 = target_Q.detach()-current_Q1
        td_error2 = target_Q.detach()-current_Q2
        critic1_loss = 0.5* (td_error1.pow(2)*weights).mean()
        critic2_loss = 0.5* (td_error2.pow(2).to(self.device)*weights).mean()
        prios = abs(((td_error1 + td_error2)/2.0 + 1e-5).squeeze())

        critic_loss = critic1_loss + critic2_loss

        writer.add_scalar('Loss critic', critic_loss, step)
        # if step % self.log_interval == 0:
        #     L.log('train_critic/loss', critic_loss, step)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # self.critic.log(L, step)
        return prios

    def update_actor_and_alpha(self, obs, pose, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, pose, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pose, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        # if step % self.log_interval == 0:
            # L.log('train_actor/loss', actor_loss, step)
            # L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        # if step % self.log_interval == 0:                                    
            # L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        writer.add_scalar('Loss actor', actor_loss, step)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()

        writer.add_scalar('Loss alpha', alpha_loss, step)
        # if step % self.log_interval == 0:
            # L.log('train_alpha/loss', alpha_loss, step)
            # L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_cpc(self, obs_anchor, obs_pos, cpc_kwargs, step):
        
        z_a = self.CURL.encode(obs_anchor)
        z_pos = self.CURL.encode(obs_pos, ema=True)
        
        logits = self.CURL.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)
        
        self.encoder_optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()
        loss.backward()

        writer.add_scalar('Loss curl', loss, step)

        self.encoder_optimizer.step()
        self.cpc_optimizer.step()
        # if step % self.log_interval == 0:
            # L.log('train/curl_loss', loss, step)


    def update(self, replay_buffer, step, writer):
        if self.encoder_type == 'pixel':
            obs, pose, action, reward, next_obs, next_pose, not_done, cpc_kwargs, idx, weights = replay_buffer.sample_cpc()
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()

        # print('---------------')
        # print(cpc_kwargs["obs_anchor"].size())
        # print(action.size())
    
        # if step % self.log_interval == 0:
            # L.log('train/batch_reward', reward.mean(), step)

        prios = self.update_critic(obs, pose, action, reward, next_obs, next_pose, not_done, step, weights, writer)

        replay_buffer.update_priorities(idx, prios.data.cpu().numpy())

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, pose, step)

        if step % self.critic_target_update_freq == 0:
            soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )
        
        if step % self.cpc_update_freq == 0 and self.encoder_type == 'pixel':
            obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
            self.update_cpc(obs_anchor, obs_pos,cpc_kwargs, step)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/SAC_model/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/SAC_model/critic_%s.pt' % (model_dir, step)
        )

    def save_curl(self, model_dir, step):
        torch.save(
            self.CURL.state_dict(), '%s/SAC_model/curl_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/SAC_model/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/SAC_model/critic_%s.pt' % (model_dir, step))
        )
        self.CURL.load_state_dict(
            torch.load('%s/SAC_model/curl_%s.pt' % (model_dir, step))
        )

def make_agent(obs_shape, action_shape, device):
    # if args.agent == 'curl_sac':
    return CurlSacAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        hidden_dim=1024,
        discount=0.99,
        init_temperature=0.1,
        alpha_lr=1e-4,
        alpha_beta=0.5,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.01,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.05,
        num_layers=4,
        num_filters=32,
        log_interval=100,
        detach_encoder=False,
        curl_latent_dim=170

    )
    # else:
    #     assert 'agent is not supported: %s' % args.agent


#----------------------------------------------------------
ACTION_V_MIN = -0.3 # m/s
ACTION_W_MIN = -0.3 # rad/s
ACTION_V_MAX = 0.3 # m/s
ACTION_W_MAX = 0.3 # rad/s
replay_buffer_size = 100000
#****************************
is_training = True
#----------------------------------------
def action_unnormalized(action, high, low):
    action = low + (action + 1.0) * 0.5 * (high - low)
    action = np.clip(action, low, high)
    return action
#**********************************

writer = SummaryWriter(dirPath + '/evaluations/',flush_secs=1, max_queue=1)

def evaluate(num_episodes=10000, encoder_type='pixel', image_size=100):
    all_ep_rewards = []
    print('evau')

    def run_eval_loop(sample_stochastically=True):

        prefix = 'stochastic_' if sample_stochastically else ''

        if sample_stochastically:
            print('stochastically evaluation')
        else:
            print('deterministic evaluation')

        for i in range(num_episodes):
            obs, obs_pixel, pose = env.reset()
            pose = np.array(pose)
            # print('-print aqui---------------------')
            done = False
            episode_reward = 0
            step = 0
            while not done:
                # if encoder_type == 'pixel':
                    # obs = center_crop_image(obs, image_size)
                with eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs_pixel, pose)
                        unnorm_action = np.array([
                            action_unnormalized(action[0], ACTION_V_MAX, 0), 
                            action_unnormalized(action[1], ACTION_W_MAX, ACTION_W_MIN),
                            action_unnormalized(action[2], ACTION_V_MAX, ACTION_V_MIN)
                            ])
                    else:
                        action = agent.select_action(obs_pixel, pose)
                        unnorm_action = np.array([
                            action_unnormalized(action[0], ACTION_V_MAX, 0), 
                            action_unnormalized(action[1], ACTION_W_MAX, ACTION_W_MIN),
                            action_unnormalized(action[2], ACTION_V_MAX, ACTION_V_MIN)                            
                            ])
                obs, obs_pixel, pose, reward, done = env.step(unnorm_action)
                pose = np.array(pose)
                # print('pose', pose)

                episode_reward += reward
                step +=1
                if step > 2:
                    print('final', i, step)
                    done = True
                    break

            all_ep_rewards.append(episode_reward)

        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)

        print('Evaluation: Mean, Best')
        print(mean_ep_reward, best_ep_reward)

        return mean_ep_reward, best_ep_reward

    return run_eval_loop(sample_stochastically=False)


if __name__ == '__main__':
    rospy.init_node('curl_node')
    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    result = Float32()
    env = Env()

    action_shape = 3
    obs_shape = (3*3, 100, 100)
    pre_aug_obs_shape = (3*3, 100, 100)

    episode, episode_reward, done = 0, 0, True
    max_steps = 1000000
    initial_step = 0
    # initial_step = 25000
    save_model_replay = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    replay_buffer = ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=replay_buffer_size,
        batch_size=170,
        device=device,
        image_size=100,
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device
    )

    print('Path: ' + dirPath)

    # agent.load(dirPath, initial_step) # 5 + 149 + 30
    # replay_buffer.load(dirPath + '/replay_memory/')

    for step in range(initial_step, max_steps):
        
        if step % 1000 == 0:
            print('start_eval')
            mean, best = evaluate()
            writer.add_scalar('Reward mean', mean, step)
            writer.add_scalar('Reward best', best, step)
            if save_model_replay:
                if step%(1*1000) == 0:
                    agent.save(dirPath, step)
                    agent.save_curl(dirPath, step)
                    # replay_buffer.save(dirPath + '/replay_memory/')
                    print('saved model and replay memory', step)
            # obs = env.reset()
            done = True
            save_model_replay = True
        
        if done:
            
            print("*********************************")
            print('Episode: ' + str(episode) + ' training')
            print('Step: ' + str(step) + ' training')
            print('Reward average per ep: ' + str(episode_reward))
            print("*********************************")

            obs, obs_pixel, pose = env.reset()
            pose = np.array(pose)

            done = False
            episode_reward = 0
            episode += 1

        if step < 200: #1000
            print('### Collecting memory ###')
            action = np.array([
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1)
                ]) 
            unnorm_action = np.array([
                action_unnormalized(action[0], ACTION_V_MAX, 0), 
                action_unnormalized(action[1], ACTION_W_MAX, ACTION_W_MIN),
                action_unnormalized(action[2], ACTION_V_MAX, ACTION_V_MIN)
                ])
        else:
            with eval_mode(agent):
                action = agent.sample_action(obs_pixel, pose)
                # print('action1 ', unnorm_action)
                unnorm_action = np.array([
                    action_unnormalized(action[0], ACTION_V_MAX, 0), 
                    action_unnormalized(action[1], ACTION_W_MAX, ACTION_W_MIN),
                    action_unnormalized(action[2], ACTION_V_MAX, ACTION_V_MIN)
                    ])

        if step >= (initial_step + 400): #1000
            num_updates = 1 
            for _ in range(num_updates):
                # print('update')
                agent.update(replay_buffer, step, writer)

        next_obs, next_obs_pixel, next_pose, reward, done = env.step(unnorm_action)
        next_pose = np.array(next_pose)
        # print('----')
        # print(next_obs_pixel.max(), next_obs.max())
        # print('action ', unnorm_action)
        # print('pose ', next_pose)
        # print('reward ', reward)
        # print('----')
        # print('state ', obs.shape)

        episode_reward += reward
        replay_buffer.add(obs_pixel, obs, pose, action, reward, next_obs_pixel, next_pose, done)
        # print('------------------')
        # rgb = np.dstack((obs_pixel[0], obs_pixel[1], obs_pixel[2]))
        # print(rgb.shape)

        # cv2.imwrite(dirPath + '/observation_rgb.png', rgb)

        # rgb = np.dstack((obs[0], obs[1], obs[2]))
        # print(rgb.shape)

        # cv2.imwrite(dirPath + '/observation_rgb_depth.png', rgb)

        if reward <= -1.:
            print('\n----collide-----\n')
            # for _ in range(1):
            #     # print('aqui2')
            #     replay_buffer.add(obs, action, reward, next_obs, done)

        obs = copy.deepcopy(next_obs)
        obs_pixel = copy.deepcopy(next_obs_pixel)
        pose = copy.deepcopy(next_pose)
        # episode_step += 1