import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy
from pynput import keyboard
import pygame

# Gym is an OpenAI toolkit for RL
import gymnasium
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros


def main():
    #from tensordict import TensorDict
    #from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

    # Initialize Pygame
    pygame.init()

    # Initialize Super Mario environment (in v0.26 change render mode to 'human' to see results on the screen)
    if gymnasium.__version__ < '0.26':
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
    else:
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='human', apply_api_compatibility=True)

    # Limit the action-space to
    #   0. walk right
    #   1. jump right
    env = JoypadSpace(env, [["A"], ["left"], ["B"], ["right"], ["right", "A"], ["left", "A"], ["NOOP"]])

    env.reset()
    next_state, reward, done, trunc, info = env.step(action=0)
    print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

    #import keyboard

    #from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    #env = JoypadSpace(env, [["A"], ["left"], ["B"], ["right"]])

    # TODO: Start with Double Q-Learning
    # TODO: Afterwards implement Dueling DQN

    # TODO: output from environment
    #(240, 256, 3),
    #0.0,
    #False,
    #{'coins': 0, 'flag_get': False, 'life': 2, 'score': 0, 'stage': 1, 'status': 'small', 'time': 400, 'world': 1, 'x_pos': 40, 'y_pos': 79}

    # TODO: Write a method for double q learning
    double_q_learning()

    testRun(env)


def double_q_learning():
    pass


# Test with key inputs
def testRun(env):
    while True:

        #if done:
        #   state = env.reset()
        #action = env.action_space.sample()

        action = 6

        #while action not in env._action_map:
            #action = env.action_space.sample()
        # Handle key press events
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            env.close()
            pygame.quit()
            return
        if keys[pygame.K_w] and keys[pygame.K_d]:
            action = 4
        elif keys[pygame.K_w] and keys[pygame.K_a]:
            action = 5
        elif keys[pygame.K_w]:
            action = 0
        elif keys[pygame.K_a]:
            action = 1
        elif keys[pygame.K_s]:
            action = 2
        elif keys[pygame.K_d]:
            action = 3



        state, reward, done, trunc, info = env.step(action=action)

        # Set action back to 0 for the next loop iteration
        #action = 3

        if done:
            state = env.reset()

        env.render()




if __name__ == "__main__":
    main()