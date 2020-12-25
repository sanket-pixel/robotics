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

def sensor_model(observation, belief, world):
    white_probability = np.array([0.1, 0.7])
    black_probability = np.array([0.9, 0.3])
    if observation == 0:
        map_probability = np.where(world == 0, black_probability[0], black_probability[1])
    elif observation == 1:
        map_probability = np.where(world == 1, white_probability[1], white_probability[0])
    corrected_belief = map_probability*belief
    return corrected_belief/np.sum(corrected_belief)



def recursive_bayes_filter(actions, observations, belief, world):

    for index in range(len(actions)):
        current_observation = observations[index+1]
        current_action = actions[index]
        predicted_belief = motion_model(current_action, belief)
        belief = sensor_model(current_observation, predicted_belief, world)
        plot_belief(belief)

    return belief


