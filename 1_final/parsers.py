#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Contains parsers for
- Dynophore JSON file, which contains statistis on the occurrences and distances of superfeatures
  and environmental partners
- Dynophore PML file, which contains 3D coordinates for points in superfeature
  point clouds
"""

import xml.etree.ElementTree as ET
import numpy as np
import compute
import os
import json

def pml_to_dict(dyno_path, n_drop = 0):
    """
    A funtion to parse dynophore PML file and write into a dict.
    Refered to dynophore code: https://github.com/wolberlab/dynophores
    Input:
        dyno_path: str - Path to dynophore output folder e.g. "./dynophore_out_2022-06-09_00-15-16-HIV/"
    Output:
        dynophore_dict: dict 
                        e.g:
                        - <superfeature id>
                          - id : str
                          - color : str
                          - center : numpy.array
                          - points : list
                            - x : float
                            - y : float
                            - z : float
                            - frame_ix : int
                            - weight : float
    Attribute:
        n_drop: int - Currently not in use. For dropping first few frames of MD trajectory
    """
    datanames = os.listdir(dyno_path)
    pml_ls = []
    for data in datanames:
        if os.path.splitext(data)[1] == ".pml":
            pml_ls.append(data)
    if len(pml_ls) == 1:
        pml_path = f"{dyno_path}{pml_ls[0]}"
    else:
        raise Exception(f"There are {len(pml_ls)} pml file in {dyno_path}. Please specify the pml_path.")
            
    dynophore_xml = ET.parse(pml_path)
    dynophore_dict = {}

    feature_clouds = dynophore_xml.findall("featureCloud")
    for feature_cloud in feature_clouds:

        # Superfeature ID
        superfeature_feature_name = feature_cloud.get("name")
        superfeature_atom_numbers = feature_cloud.get("involvedAtomSerials")
        superfeature_id = f"{superfeature_feature_name}[{superfeature_atom_numbers}]"
        # Superfeature color
        superfeature_color = feature_cloud.get("featureColor")
        # Superfeature cloud center
        center = feature_cloud.find("position")
        center_data = np.array(
            [
                float(center.get("x3")),
                float(center.get("y3")),
                float(center.get("z3")),
            ]
        )
        # Superfeature cloud points
        additional_points = feature_cloud.findall("additionalPoint")
        additional_points_data = []
        for additional_point in additional_points:
            frame_ix = int(additional_point.get("frameIndex"))
            if frame_ix >= n_drop:
                additional_point_data = {
                    "x": float(additional_point.get("x3")),
                    "y": float(additional_point.get("y3")),
                    "z": float(additional_point.get("z3")),
                    "frame_ix": frame_ix - n_drop,
                    "weight": float(additional_point.get("weight")),
                }
                additional_points_data.append(additional_point_data)

        dynophore_dict[superfeature_id] = {}
        dynophore_dict[superfeature_id]["id"] = superfeature_id
        dynophore_dict[superfeature_id]["color"] = superfeature_color
        dynophore_dict[superfeature_id]["center"] = center_data
        dynophore_dict[superfeature_id]["points"] = additional_points_data

    return dynophore_dict


def extract_coordinates(dyno_dict, feature_key):
    """A function to extract xyz-trajectory for one superfeature into a NumPy array
       Input:
           dyno_dict: dict - obtained from pml_to_dict()
           feature_key: str - superfeature name
       Output:
           coordinates: ndarray - shape (n, 3)
    """
    coordinates = []
    last_frame = None
    for p in dyno_dict[feature_key]['points']:
        if p["frame_ix"] == last_frame:
            continue
        last_frame = p["frame_ix"]
        coordinates.append([p["x"], p["y"], p["z"]])
    return np.asarray(coordinates)


def extract_time(dyno_dict, feature_key):
    """A funtion to extract frame id for each feature.
       Input:
           dyno_dict: dict - obtained from pml_to_dict()
           feature_key: str - superfeature name
       Output:
           frames: ndarray - shape (n, 1)
    """
    time = []
    last_frame = None
    for p in dyno_dict[feature_key]['points']:
        if p["frame_ix"] == last_frame:
            continue
        last_frame = p["frame_ix"]
        time.append(p["frame_ix"])
    return np.asarray(time)


def extract_norm(dyno_dict, env_partner_dict):
    '''Normalize xyz coordinates and add frame information.
       Input:
           dyno_dict: dict - obtained from pml_to_dict()
           env_partner_dict: dict - contains info of environment partners for each superfeature. Currently not in use
       Output:
           data: dict
                 e.g:
                 - <superfeature id>: str
                   - points : ndarray after normalization
                     - x : float
                     - y : float
                     - z : float
                   - frames : ndarray
                   - non_norm : ndarray without normalization
                   - color : str
                   - id : int
                   - env_partner : dict
                     - residue name
                       - atom number in pdb
       '''
    data = {}
    for key in dyno_dict.keys():
        points_tmp = extract_coordinates(dyno_dict, key)
        if points_tmp != []:
            points_min, points_max = np.min(points_tmp), np.max(points_tmp)
            norm_points = (points_tmp-points_min)/(points_max-points_min)
            frames = extract_time(dyno_dict, key)
            data[key] = {
                "points": norm_points,
                "frames": frames,
                "non_norm": points_tmp,
                "color": dyno_dict[key]["color"],
                "id": dyno_dict[key]["id"],
                "env_partner": env_partner_dict[key]
            }
        else:
            frames = extract_time(dyno_dict, key)
            data[key] = {
                "points": np.array([0, 0, 0]).reshape(-1, 1),
                "frames": frames,
                "non_norm": np.array([0, 0, 0]).reshape(-1, 1)
            }
    return data


def get_env_partner(dyno_path):
    """A funtion to parse JSON file.
       Input:
           dyno_path: str - Path to dynophore output folder e.g. "./dynophore_out_2022-06-09_00-15-16-HIV/"
       Output:
           env_partner_dict: dict
           e.g:
           - <superfeature id> : str
             - residue name : str
               - invoved atom number in pdb file : list
    """
    datanames = os.listdir(dyno_path)
    json_ls = []
    for data in datanames:
        if os.path.splitext(data)[1] == ".json":
            json_ls.append(data)
    if len(json_ls) == 1:
        json_path = f"{dyno_path}{json_ls[0]}"
    else:
        raise Exception(f"There are {len(json_ls)} json file in {dyno_path}. Please specify the json_path.")
        
    with open(json_path, "r") as f:
        json_string = f.read()
        dynophore_dict = json.loads(json_string)
        
    superfeature_data = dynophore_dict["superfeatures"]
    env_partner_dict = {}
    for i in range(len(superfeature_data)):
        key = superfeature_data[i]["id"]
        env_partner = {}
        env_partner_data = dynophore_dict["superfeatures"][i]["envpartners"]
        for i in range(len(env_partner_data)):
            env_partner_nr = env_partner_data[i]["atom_numbers"]
            env_partner_name = env_partner_data[i]["id"]
            env_partner[env_partner_name] = env_partner_nr
        env_partner_dict[key] = env_partner

    return env_partner_dict


def pre_process(dyno_path, include_time = False, n_drop = 0):
    '''A fundtion for combined data pre-processing work flow
       Inputs:
           dyno_path: str - Path to dynophore output folder e.g. "./dynophore_out_2022-06-09_00-15-16-HIV/"
       Attributes:
           include_time: bool - True for including time distance
           n_drop: int - Currently not in use. For dropping first few frames of MD trajectory
       Output:
           data: dict
                 e.g:
                 - <superfeature id>: str
                   - points : ndarray after normalization
                     - x : float
                     - y : float
                     - z : float
                   - frames : ndarray
                   - non_norm : ndarray without normalization
                   - distances : ndarray
                   - color : str
                   - id : int
                   - env_partner : dict
                     - residue name
                       - atom number in pdb
                   
    '''
    dynophore_dict = pml_to_dict(dyno_path, n_drop = n_drop)
    env_partner_dict = get_env_partner(dyno_path)
    data = extract_norm(dynophore_dict, env_partner_dict)
    data = compute.add_distance_mat(data, include_time = include_time)
    max_frame = compute.get_max_frame(data)
    print(f"Data pre-processed: {max_frame} frames in trajectory")
    
    return data

def get_frame_pose_map(data, model):
    '''A funtion to extract information on each frame belong to which binding state
       Input:
           data: dict
           model: KMedoids model object -  e.g. KMedoids(method='pam', metric='manhattan', n_clusters=3)
       Output:
           frame_pose_map: dict - frame_number: binding_state_number
    '''
    cluster_frames_map = compute.get_frames_each_cluster(model)
    frame_pose_ls = [data_ for data_ in cluster_frames_map.values()]
    max_frame = compute.get_max_frame(data)

    frame_pose_map = {}

    for frame in range(max_frame+1):
        for pose_nr in range(len(frame_pose_ls)):
            if frame in frame_pose_ls[pose_nr]:
                frame_pose_map[frame] = pose_nr

    return frame_pose_map

def get_wrap_data(data, model):
    '''
    wrap up information for all clusters of all superfeatures for visualization
    Input:
        data: dict
        model: KMedoids model object -  e.g. KMedoids(method='pam', metric='manhattan', n_clusters=3)
    Output:
        wrap_data: list
                   e.g: [x, y, z, (frame), label in each superfeature, superfeature_nr, binding_state_nr]
    '''
    wrap_data = []
    cluster_frames_map = compute.get_frames_each_cluster(model)
    frame_pose_map = get_frame_pose_map(data, model)

    for i, key in enumerate(data.keys()):
        final_data = data[key]["non_norm"]
        feature_frames = data[key]["frames"]
        cluster_temp = data[key]["clustering"]
        label = cluster_temp._labels.labels
        superfeature_nr = np.array([i] * len(final_data))
        
        binding_state_nr = []
        for frame in feature_frames:
            binding_state_nr_tmp = frame_pose_map[frame]
            binding_state_nr.append(binding_state_nr_tmp)
        binding_state_nr = np.array(binding_state_nr)

        final_data = np.column_stack((final_data, label))
        final_data = np.column_stack((final_data, superfeature_nr, binding_state_nr))
        
        wrap_data.append(final_data)
    
    return wrap_data


def get_feature_name(feature):
    '''A function to extract feature type
       Input:
           feature: str - superfeature name like "H[3185,3179,3187,3183,3181,3178]"
       Output:
           feature type: str - e.g. "H" "HBD"
           '''
    for i in range(len(feature)):
        if feature[i] == '[':
            return feature[:i]