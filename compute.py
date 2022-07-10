#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.metrics import pairwise_distances
from scipy.signal import argrelmax, argrelmin
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import MDAnalysis as mda

import visualize, parsers, clustering


def add_distance_mat(data, include_time = False, time_prop = 1/3):
    '''A funtion to add distance matrix
       Input:
           data: dict
       Output:
           data: dict - data with distance matrix added
       Attribute:
           include_time: bool - True for including time distance in clustering step
           time_prop: float - temporal contribution to distance metric in CNNclustering
       '''
    max_frame = get_max_frame(data)
    for key in data.keys():
        data_tmp = data[key]["points"]
        if len(data_tmp.shape) == 1:
            data_tmp.reshape(-1, 1)
        distances = pairwise_distances(data_tmp)
        points_min, points_max = np.min(distances), np.max(distances)
        distances = (distances-points_min)/(points_max-points_min)
        if include_time:
            # add time component in distance matrix
            frames_distance = pairwise_distances(data[key]["frames"].reshape(-1,1))
            frames_distance = np.sqrt(frames_distance)
            frame_min, frame_max = np.min(frames_distance), np.max(frames_distance)
            frames_distance = (frames_distance-frame_min)/(frame_max-frame_min)
            
            distances = distances + time_prop*frames_distance
            dist_min, dist_max = np.min(distances), np.max(distances)
            distances = (distances-dist_min)/(dist_max-dist_min)

        data[key]["distances"] = distances
    return data


def add_auto_param(data, include_time = False, info_table = True, 
                   frequency_cutoff = 0.06, plot = True):
    '''A funtion to offer preliminary prediction on radius_cutoff and cnn_cutoff 
       based on distance distribution of data points
       Input:
           data: dict
       Output:
           data: dict - data with parameters added
           e.g. of parameter
           ["params"] = {
                        "radius_cutoff": 0.29,
                        "cnn_cutoff": 10,
                        "member_cutoff": 10
                        }
       Attributes:
           include_time: bool - True for including time distance
           info_table: bool - True for print parameter chosen and predict cluster features of each superfeature
           frequency_cutoff: float - A threshold for superfeature frequency, under this threshold, no parameter will be predicted
           plot: bool - True for plotting the distance histogram
       '''
    default_cluster_params = {
        "radius_cutoff": 0.2,
        "cnn_cutoff": 5,
        "member_cutoff": 10
    }
        
    print("*"*100)
    print("Start adding parameter automatically. For feature with fewer points or noisy data, default parameters will be added.")
    print(default_cluster_params)
    print("*"*100)
    max_frame = get_max_frame(data)
    for i, key in enumerate(data.keys()):
        if not include_time:
            data_tmp = data[key]["points"]
            distances = pairwise_distances(data_tmp)
            
        else:
            '''include time'''
            distances = data[key]["distances"]

        radius_multi = 0.8 # multiplier used in radius calculation
        cluster_benchmark = 0
        min_freq = 0
        frame_nr = len(distances[0])
        
        radius = 0
        
        weights = np.zeros_like(distances.flatten()) + 1. / distances.flatten().size
        y, x, _ = plt.hist(distances.flatten(), bins=100, color='y', weights=weights)
        plt.close()

        if frame_nr >= frequency_cutoff*max_frame:
            n, bins = np.histogram(distances, 15)

            # get the peaks and valleys' position
            x_left = np.min(distances)
            x_right = np.max(distances)
            interval = (x_right - x_left)/15

            # predict min cluster number based on distance distribution
            n_cluster = len(argrelmax(n)[0])
            if n_cluster <= 4: # cluster > 4 could resulted from noice data, so not predicted
                # get minimal frequency at valleys
                valley_pos = [i*5 for i in argrelmin(n)]
                if len(y[tuple(valley_pos)]):
                    min_freq = round(min(y[tuple(valley_pos)]), 3)
                    max_freq = round(np.max(y), 3)

                cluster_benchmark = n_cluster

                max_ = (np.array(argrelmax(n))*interval)[0]
                min_ = (np.array(argrelmin(n))*interval)[0]
                min_ = np.insert(min_, 0, 0)

                # predict radius based on average peak width/2
                min_len = min(len(max_), len(min_))
                radius = round(np.mean(max_[:min_len] - min_[:min_len])*radius_multi, 3)

                # predict cnn_cutoff based on points number: 5%*points number
                if min_freq != 0:
                    cnn_cutoff_ = round(min(frame_nr*min_freq, frame_nr*0.05, max_frame*0.003))
                else:
                    cnn_cutoff_ = round(min(frame_nr*0.05, max_frame*0.003))

                data[key]["params"] = {
                    "radius_cutoff": radius,
                    "cnn_cutoff": cnn_cutoff_,
                    "member_cutoff": 10
                }
            else:
                data[key]["params"] = default_cluster_params
        else:
            data[key]["params"] = default_cluster_params

        data[key]["cluster_benchmark"] = cluster_benchmark
        if plot:
            value = data[key]["params"]["radius_cutoff"]
            plt.figure(figsize=(4,2))
            plt.hist(distances.flatten(), bins=100, color='y')
            plt.axvline(value)

            plt.title(f"{i}: {key}", fontsize = 6)
            plt.annotate(f"r = {round(value, 2)}", (0.05, 0.95), xycoords="axes fraction", fontsize=6)
            plt.show()
        # print infor table
        if info_table:
            print(i, key)
            print("-"*52)
            print ("{:<15} {:<15} {:<8} {:<15}".format('min_frequency',
                                                       'cluster_benchmark','radius','cnn_cutoff')) # ti
            print ("{:<15} {:<15} {:<8} {:<15}".format(min_freq, cluster_benchmark, data[key]["params"]["radius_cutoff"], data[key]["params"]["cnn_cutoff"]))
            print("-"*52)
        print(f"Parameter added for {key}")
    return data

def get_max_frame(data):
    '''A function to get maximal frame number of analyzed MD trajectory
       Input:
           data: dict
       Output:
           max_frame: int
       '''
    return max([x["frames"][-1] for x in data.values()])
    
def get_state_matrix(data):
    '''wrap up state information of all superfeatures into one ndarray
       Input:
           data: dict
       Output: 
           state_matrix: ndarray - shape (max_frame_nr, superfature_count)
       '''
    state_matrix = []
    max_frame = get_max_frame(data)

    for fkey in data.keys():
        current_frame = 0
        state_matrix.append([])
        padded = state_matrix[-1]
        for frame_id, clabel in zip(data[fkey]["frames"], data[fkey]["clustering"].labels):

            while frame_id > current_frame:
                padded.append(0)
                current_frame += 1
            padded.append(clabel)
            current_frame += 1
        while current_frame <= max_frame:
            padded.append(0)
            current_frame += 1
            
    state_matrix = np.asarray(state_matrix).T
        
    return state_matrix


def get_one_hot_encoding(state_matrix):
    '''A funtion to transform encode state_matrix in one-hot-key way
       Input:
           state_matrix: ndarray - a array contain all clustered state information of all superfeatures
       Output: 
           one_hot_matrix: ndarray
           e.g.:
           frame | existance of state 0 in superfeature 1 | existance of state 1 in superfeature 1...
           1     | 1 (means exist)                        | 0 (means absence)
           2     ...
           3     ...
    '''
    encoder = OneHotEncoder(sparse = False)
    one_hot_matrix = encoder.fit_transform(state_matrix)
    return one_hot_matrix


def get_adjusted_one_hot_encoding(data, one_hot_matrix):
    '''A funtion to add one more colomn for each superfeature to show the effective existence of superfeature,
       effective existence = interaction exists AND belongs to non-noise cluster.
       0 for no effective existence, 1 for the opposite.
       Aim to increase (double) the weight in difference between exxiting and non-existing frame regarding one superfeature
       Inputs:
           data: dict
           one_hot_matrix: ndarray - a matrix contain one hot key encoding. Shape of (n, n)
       Output:
           adjusted_one_hot_matrix: ndarray - with width >= one_hot_matrix
       '''
    no_interact_col = []
    state_matrix = get_state_matrix(data)
    states_per_interaction = clustering.get_states_per_interaction(state_matrix)
    for pos in range(0, len(one_hot_matrix[0])):
        superfeature_idx, state = clustering.get_feature_state_from_onehot_position(pos, states_per_interaction)
        if state == 0:
            no_interact_col.append(pos)
    adjusted_one_hot_matrix = np.hstack((one_hot_matrix, 1 - one_hot_matrix[:, no_interact_col]))
    return adjusted_one_hot_matrix
    
def get_frames_each_cluster(model):
    '''A funtion to extract frames within each binding pose
       Input:
           model: KMedoids model object -  e.g. KMedoids(method='pam', metric='manhattan', n_clusters=3)
       Output: 
           cluster_frames_map: dict
           e.g.:
           {binding state 1: ndarray of frames,
            binding state 1: ...}
            '''
    state_nr = np.max(model.labels_) + 1
    cluster_frames_map = {
        k: np.where(model.labels_ == k)[0]
        for k in range(state_nr)
    }
    return cluster_frames_map


def get_state_statistis(data, model):
    '''Get frame count of each superfeature state for each binding poses
       Input:
           data: dict
           model: KMedoids model object -  e.g. KMedoids(method='pam', metric='manhattan', n_clusters=3)
       Output: 
           state_statistis: dict
           e.g. {binding pose 1: {superfeature 1:{feature state 0: count of frames, feature state 1: count of frames},
                                 {superfeature 2:{feature state 0: count of frames},
                                 {superfeature 3...}}}'''
    state_matrix = get_state_matrix(data)
    cluster_frames_map = get_frames_each_cluster(model)
    state_statistis = {}

    for state_idx, frames in cluster_frames_map.items():
        state_data = {}
        data_per_state = state_matrix[frames]
        for feature_idx, feature in enumerate(data.keys()):
            data_per_feature_state = data_per_state[:, feature_idx]
            stata_feature_data = {}
            for cluster in np.unique(data_per_feature_state):
                cluster_count = len(data_per_feature_state[data_per_feature_state == cluster])
                stata_feature_data[cluster] = cluster_count
            state_data[feature_idx] = stata_feature_data

        state_statistis[state_idx] = state_data
        
    return state_statistis
    
    
def get_feature_freq_per_state(data, model):
    '''Get frequency of each superfeature in each binding pose
       Input:
           data: dict
           model: KMedoids model object -  e.g. KMedoids(method='pam', metric='manhattan', n_clusters=3)
       Output: 
           feature_freq_per_state: dict
           e.g. {binding pose 1: {superfeature 1:frequency,
                                 {superfeature 2:frequency,
                                 {superfeature 3...}}}'''
    
    state_statistis = get_state_statistis(data, model)
    max_frame = get_max_frame(data)
    
    feature_freq_per_state = {}
    for state_idx, data_ in state_statistis.items():
        cache = {}
        for feature, _data_ in data_.items():
            number = 0
            for state, count in _data_.items():
                if state > 0:
                    number += count
            cache[feature] = number/max_frame
        feature_freq_per_state[state_idx] = cache
    return feature_freq_per_state


def get_interact_summary(data, model):
    '''A funtion to print interaction summary for all binding states
       Frequency of each superfeature within each cluster is shown.
       Input:
           data: dict
           model: KMedoids model object -  e.g. KMedoids(method='pam', metric='manhattan', n_clusters=3)
       Output: 
           interact_summary: DataFrame
           e.g.
                state0     state1     state2     state3      superfeature
           0   0.077521   0.130609   0.067970   0.290982   H[3187,3181,3178,3179,3185,3183]
           1   0.000000   0.000000   0.000000   0.000000   H[3146]
           2   0.249000   0.117503   0.166148   0.432474   H[3150,3152,3158,3156,3154,3149]
           '''
    feature_per_state = get_feature_freq_per_state(data, model)
    feature_per_state_df = pd.DataFrame(feature_per_state)
    interact_summary = feature_per_state_df.copy()
    interact_summary["feature"] = data.keys()
    return interact_summary

   
def get_geo_center(data):
    '''A funtion to obtain geographic center of data points
       Input:
           data: ndarray with first three columns as x, y, z
    '''
    x, y, z = np.mean(data[:, 0]), np.mean(data[:, 1]), np.mean(data[:, 2])
    return (x, y, z)

#################################### NOT IN USE ##############################################

def reduce_frames(pdb_path, dcd_path, select = "protein or chainID X", out_path = f"output/reduced.dcd", final_n_frame = 500):
    '''A funtion to write reduce trajectory to desired length in output folder
       Used for plot_whole_rmsd() and plot_diffusionmap()
       Currently not in use.'''
    u = mda.Universe(pdb_path, dcd_path)
    frames = [i for i in range(0, len(u.trajectory), len(u.trajectory)//final_n_frame)]
    n_atoms = int(str(u)[-11:-7])
    
    with mda.Writer(out_path, n_atoms = n_atoms) as W:
        for ts in u.trajectory[frames]:
            W.write(u.select_atoms(select))
            