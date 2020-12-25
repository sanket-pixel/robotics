#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plot_belief(belief):
    
    plt.figure()
    
    ax = plt.subplot(2,1,1)
    ax.matshow(belief.reshape(1, belief.shape[0]))
    ax.set_xticks(np.arange(0, belief.shape[0],1))
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks([])
    ax.title.set_text("Grid")
    
    ax = plt.subplot(2, 1, 2)
    ax.bar(np.arange(0, belief.shape[0]), belief)
    ax.set_xticks(np.arange(0, belief.shape[0], 1))
    ax.set_ylim([0, 1.05])
    ax.title.set_text("Histogram")


def motion_model(action, belief):
    motion_probabiity = np.array([[0.7, 0.2, 0.1]])
    shifted_belief = np.stack([np.roll(np.pad(belief,1), 1)[1:16], belief, np.roll(np.pad(belief,1), -1)[1:16]])
    if action == 1 :
        return np.dot(motion_probabiity, shifted_belief)[0]
    elif action == -1 :
        return  np.dot(np.fliplr(motion_probabiity), shifted_belief)[0]

# def sensor_model(observation, belief, world):
    # add code here

# def recursive_bayes_filter(actions, observations, belief, world):
    # add code here

