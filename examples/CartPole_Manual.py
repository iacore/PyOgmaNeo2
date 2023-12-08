# ----------------------------------------------------------------------------
#  PyOgmaNeo
#  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of EOgmaNeo is licensed to you under the terms described
#  in the PYEOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

# Simple Cart-Pole example

import pyogmaneo
import gym
import numpy as np

# Create the environment
env = gym.make('CartPole-v1')

# Get observation size
numObs = env.observation_space.shape[0] # 4 values for Cart-Pole
numActions = env.action_space.n # N actions (1 discrete value)

# Define binning resolution
obsColumnSize = 32

# Set the number of threads
pyogmaneo.ComputeSystem.setNumThreads(4)

# Create the compute system
cs = pyogmaneo.ComputeSystem()

# Define layer descriptors: Parameters of each layer upon creation
lds = []

for i in range(2): # Layers with exponential memory. Not much memory is needed for Cart-Pole, so we only use 2 layers
    ld = pyogmaneo.LayerDesc()

    # Set the hidden (encoder) layer size: width x height x columnSize
    ld.hiddenSize = pyogmaneo.Int3(4, 4, 16)

    ld.ffRadius = 4 # Sparse coder radius onto visible layers
    ld.pRadius = 4 # Predictor radius onto sparse coder hidden layer (and feed back)
    ld.aRadius = 4 # Actor radius onto sparse coder hidden layer (and feed back)

    ld.ticksPerUpdate = 2 # How many ticks before a layer updates (compared to previous layer) - clock speed for exponential memory
    ld.temporalHorizon = 4 # Memory horizon of the layer. Must be greater or equal to ticksPerUpdate
    
    lds.append(ld)

# Create the hierarchy: Provided with input layer sizes (a single column in this case), and input types (a single predicted layer)
h = pyogmaneo.Hierarchy(cs, [ pyogmaneo.Int3(1, numObs, obsColumnSize), pyogmaneo.Int3(1, 1, numActions) ], [ pyogmaneo.inputTypeNone, pyogmaneo.inputTypeAction ], lds)

max_vals = np.zeros(4) + 0.1
min_vals = np.zeros(4) - 0.1
def encode_obs(obs):
	max_vals[:] = np.maximum(obs, max_vals)
	min_vals[:] = np.minimum(obs, min_vals)
	return ((obs - min_vals) / (max_vals - min_vals) * (obsColumnSize - 1) + 0.5).astype(np.int32).tolist()

def simulate(env, max_steps):
    obs = env.reset()[0]

    predictions = h.getPredictionCs(1)

    # Timesteps
    for t in range(max_steps):
        # Bin the 4 observations. Since we don't know the limits of the observation, we just squash it
        binnedObs = encode_obs(obs)
        #(sigmoid(obs * obsSquashScale) * (obsColumnSize - 1) + 0.5).astype(np.int32).ravel().tolist()

        h.step(cs, [ binnedObs, predictions ], True, 1/max_steps)
        predictions = h.getPredictionCs(1)

        # Retrieve the action, the hierarchy already automatically applied exploration
        action = predictions[0] # First and only column

        obs, _reward, terminated, truncated, info = env.step(action)

        # punish (-1.0) when terminated
        if terminated:
            #binnedObs = (sigmoid(obs * obsSquashScale) * (obsColumnSize - 1) + 0.5).astype(np.int32).ravel().tolist()
            h.step(cs, [ binnedObs, predictions ], True, -1.0)

            print("Episode {} finished after {} timesteps".format(episode + 1, t + 1))
            break

for episode in range(10000):
    if episode%500==0:
        env = gym.make('CartPole-v1', render_mode='human')
    else:
        env = gym.make('CartPole-v1')
    simulate(env, max_steps=500)

env = gym.make('CartPole-v1', render_mode='human')
simulate(env, max_steps=2000)
env.close()
