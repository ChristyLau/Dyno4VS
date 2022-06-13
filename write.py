#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import mdtraj
import numpy as np
import visualize, compute, parsers
import xml.etree.ElementTree as et
from matplotlib import colors


def split_trajectory(model, pdb_path, dcd_path, output_directory, n_drop = 0):
    '''Write trajectory in each cluster into dcd file
       Inputs:
           model: KMedoids model object -  e.g. KMedoids(method='pam', metric='manhattan', n_clusters=3)
           pdb_path: str - Path to pdb file
           dcd_path: str - Path to MD trajectory
           output_directory: str - Path to output directory
       Attribute:
           n_drop: int - Currently not in use. For dropping first few frames of MD trajectory
    '''
    state_nr = np.max(model.labels_) + 1
    cluster_frames_map = compute.get_frames_each_cluster(model)
    trajectory = mdtraj.load_dcd(
            dcd_path,
            top = pdb_path,
            )

    cluster_traj_map = {
        cluster_id: trajectory[cluster_frames_map[cluster_id]]
        for cluster_id in range(state_nr)
    }

    for cluster_id, v in cluster_traj_map.items():
        v.save_dcd(f"{output_directory}cluster_{cluster_id}.dcd")
        print(f"Trajectory of cluster {cluster_id} written to {output_directory}")
        


def extract_superfeature(data, wrap_data, pose_nr = 0, 
                         include_time = False, frequency_cutoff = None, noise_cutoff = 0.2):
    '''
    Extract new superfeature after clustering of given binding pose. 
    Also return wrap_data according to new superfeatures for writing into LigandScout file.
    Inputs: 
        data: dict
        wrap_data: list - list of ndarray contain [center_x, center_y, center_z, radius, superfeatures_name, cluster_nr] for each frame per superfeature
    Output:
        (superfeature, points_new_sp) : tuple
            superfeature: list
                          - new feature idx: int
                          - feature_type : str
                          - Mandatory?: str
                          - origin_coord: ndarray
                          - origin_tolerance: float
                          - target_coord: ndarray
                          - target_tolerance: float
                          - weight: float
                          - color: str
                          - envPatner: dict
                          e.g: 
                          [[1, 'H', 'M', array([1.054523, 3.626301, 5.687662]), 
                          1.5, array([1.054523, 3.626301, 5.687662]), 0.0, 1.0, 
                          'ffc20e', {'ALA_28_A[444,448]': [444, 448]}]
            points_new_sp: list
                           - x : float
                           - y : float
                           - z : float
                           - (frame) : float
                           - cluster idx in static clustering : float
                           - superfeature idx from dynophore : float
    Attributes:
        pose_nr: int - idx of binding mode
        include_time: bool - True for consider time distance
        frequency_cutoff: dict - frequency_cutoff for different superfeatures 
                                 for dropping noise superfeature in certain cluster, 
                                 e.g. only 1 point in sp1 in cluster 0
        noise_cutoff: float - drop less dominent clusters within each superfeature 
                              as pharmacophone show no difference for main driver or only few points, 
                              this attribute is to keep the dominent clusters.)
    Notice/TODO:
        for HBA, HBD, AR no envPartner was analyzed.
        Tolerance is set as constant: 0.2 for HBA, HBD, 0.5 for H, NI, PI. 
        Direction of AR is meaningless, only shows the center of AR.
    '''
    if frequency_cutoff == None:
        frequency_cutoff = {'H': 0.03, 'AR': 0.005,
                            'HBD': 0.03, 'HBA': 0.03,
                            'PI': 0.03, 'NI': 0.03}
    
    col_nr = 3
    if include_time:
        col_nr = 4
        
    superfeature = []
    points_new_sp = []
    superfeatures = list(data.keys())
    max_frame = compute.get_max_frame(data)
    number = 1
    
    # for each superfeature
    for idx_sp in range(len(wrap_data)):
        color = list(data.items())[idx_sp][1]["color"]
        env_partner = list(data.items())[idx_sp][1]["env_partner"]
        superfeature_data = wrap_data[idx_sp]
        superfeature_data = superfeature_data[superfeature_data[:, -1] == pose_nr]
        superfeature_data = superfeature_data[superfeature_data[:, col_nr] != 0] # exclude noise
        
        frame_nr = len(superfeature_data)  # how many frames are not noises
        feature = parsers.get_feature_name(superfeatures[idx_sp])
        freq_cutoff = frequency_cutoff[feature]

        try:
            cluster_nr = np.unique(superfeature_data[:, col_nr])[-1]  # cluster number in one superfeature
        except:
            cluster_nr = 0
        if len(superfeature_data) > max_frame * freq_cutoff: # the frequency observed in interact_table
            while cluster_nr > 0: # do not write out noise, whose label is 0
                cluster_data = superfeature_data[superfeature_data[:, col_nr] == (cluster_nr)]
                if len(cluster_data) > frame_nr * noise_cutoff:
                    center_coord = compute.get_geo_center(cluster_data)
                    if feature in ['H', 'NI', 'PI', 'AR']:
                        radius = 1.5
                    else: # ['HBA', 'HBD']:
                        radius = 0.2
                    superfeature.append([number, feature, "M", np.array(center_coord), radius, 
                                         np.array(center_coord), 0., 1., color, env_partner, list(data.keys())[idx_sp]])
                    points_new_sp.append(cluster_data)
                    number += 1
                cluster_nr -= 1
    return (superfeature, points_new_sp)

def pml_feature_point(pharmacophore, feature, points_new_sp, combine = True):
    """ This function generates an xml branch for positive and negative ionizable features as well as hydrophobic
    interactions. 
    Code from Pyrod https://github.com/wolberlab/pyrod
    Input:
        pharmacophore: ElementTree Node - parent node in xml tree
        feature: list - contains feature information for outputting
        points_new_sp: ndarray - contains coordinates of all points within this new superfeature
        combine: bool - True for writing condense pharmacophore and point together
                        False for only writing condense pharmacophore
    """
    point_name = feature[1]
    point_featureId = '{}_{}'.format(feature[1], feature[0])
    point_optional = 'false'
    point_disabled = 'false'
    point_weight = str(feature[7])
    position_x3, position_y3, position_z3 = str(feature[3][0]), str(feature[3][1]), str(feature[3][2])
    position_tolerance = str(feature[4])
    position_attributes = {'x3': position_x3, 'y3': position_y3, 'z3': position_z3, 'tolerance': position_tolerance}
    if not combine:
        point_attributes = {'name': point_name, 'featureId': point_featureId, 'optional': point_optional,
                            'disabled': point_disabled, 'weight': point_weight}
        point = et.SubElement(pharmacophore, 'point', attrib=point_attributes)
        et.SubElement(point, 'position', attrib=position_attributes)
    else: # pharmacophore and points combined
        featureCloud_attributes = {'name': point_name, 'featureColor': feature[-3], 'featureId': point_featureId,
                                  'optional': point_optional, 'disabled': point_disabled, 'weight': point_weight,
                                  'orig_superfeature': feature[-1], 'envPatner': feature[-2]}
        featureCloud = et.SubElement(pharmacophore, 'featureCloud', attrib=featureCloud_attributes)
        et.SubElement(featureCloud, 'position', attrib=position_attributes)
        for coord in points_new_sp:
            additionalPoint = et.SubElement(featureCloud, "additionalPoint", 
                                            x3=f"{coord[0]}", y3=f"{coord[1]}", z3=f"{coord[2]}", 
                                            weight="1.0")
    return


def pml_feature_plane(pharmacophore, feature, points_new_sp, combine = True):
    """ This function generates an xml branch for aromatic interactions. 
    Code modified to Pyrod https://github.com/wolberlab/pyrod
    Input:
        pharmacophore: ElementTree Node - parent node in xml tree
        feature: list - contains feature information for outputting
        points_new_sp: ndarray - contains coordinates of all points within this new superfeature
        combine: bool - True for writing condense pharmacophore and point together
                        False for only writing condense pharmacophore
    """
    plane_name = 'AR'
    plane_featureId = '{}_{}'.format('ai', feature[0])
    plane_optional = 'false'
    plane_disabled = 'false'
    plane_weight = str(feature[7])
    position_x3, position_y3, position_z3 = str(feature[3][0]), str(feature[3][1]), str(feature[3][2])
    position_tolerance = str(feature[4])
    # As no envPartner information analyzed, set normal coordinates to (0, 0, 0) help show AR as a sphere at the center
#     normal_x3, normal_y3, normal_z3 = position_x3, position_y3, position_z3
    normal_x3, normal_y3, normal_z3 = str(0.0), str(0.0), str(0.0)
    
    normal_tolerance = str(feature[6])
    position_attributes = {'x3': position_x3, 'y3': position_y3, 'z3': position_z3, 'tolerance': position_tolerance}
    if not combine:
        plane_attributes = {'name': plane_name, 'featureId': plane_featureId, 'optional': plane_optional,
                            'disabled': plane_disabled, 'weight': plane_weight}
        plane = et.SubElement(pharmacophore, 'plane', attrib=plane_attributes)
        et.SubElement(plane, 'position', attrib=position_attributes)
        normal_attributes = {'x3': normal_x3, 'y3': normal_y3, 'z3': normal_z3, 'tolerance': normal_tolerance}
        et.SubElement(plane, 'normal', attrib = normal_attributes)
    else:
        featureCloud_attributes = {'name': plane_name, 'featureColor': feature[-3], 'featureId': plane_featureId,
                                  'optional': plane_optional, 'disabled': plane_disabled, 'weight': plane_weight,
                                  'orig_superfeature': feature[-1], 'envPatner': feature[-2]}
        featureCloud = et.SubElement(pharmacophore, 'featureCloud', attrib = featureCloud_attributes)
        et.SubElement(featureCloud, 'position', attrib=position_attributes)
        for coord in points_new_sp:
            additionalPoint = et.SubElement(featureCloud, "additionalPoint", 
                                            x3=f"{coord[0]}", y3=f"{coord[1]}", z3=f"{coord[2]}", 
                                            weight="1.0")
    return

def pml_feature_vector(pharmacophore, feature, points_new_sp, combine = True):
    """ This function generates an xml branch for hydrogen bonds. 
    Code refered to Pyrod https://github.com/wolberlab/pyrod
    Input:
        pharmacophore: ElementTree Node - parent node in xml tree
        feature: list - contains feature information for outputting
        points_new_sp: ndarray - contains coordinates of all points within this new superfeature
        combine: bool - True for writing condense pharmacophore and point together
                        False for only writing condense pharmacophore
    """
    # Firstly assume it is donor
    vector_name = 'HBD'
    vector_featureId = '{}_{}'.format(feature[1], feature[0])
    if feature[1] in ['HBD']:
        vector_featureId = '{}_{}'.format(feature[1], feature[0])
        
    vector_pointsToLigand = 'false'
    vector_hasSyntheticProjectedPoint = 'false'
    vector_optional = 'false'
    vector_disabled = 'false'
    vector_weight = str(feature[7])
    origin_x3, origin_y3, origin_z3 = str(feature[3][0]), str(feature[3][1]), str(feature[3][2])
    origin_tolerance = str(feature[4])
    target_x3, target_y3, target_z3 = [str(feature[5][0]), str(feature[5][1]),
                                       str(feature[5][2])]
    target_tolerance = str(feature[6])

    if feature[1] in ['HBA']:  # switch to acceptor
        vector_name = 'HBA'
        vector_pointsToLigand = 'true'
        origin_x3, origin_y3, origin_z3, target_x3, target_y3, target_z3 = [target_x3, target_y3, target_z3,
                                                                            origin_x3, origin_y3, origin_z3]
        origin_tolerance, target_tolerance = target_tolerance, origin_tolerance
    origin_attributes = {'x3': origin_x3, 'y3': origin_y3, 'z3': origin_z3, 'tolerance': origin_tolerance}
    target_attributes = {'x3': target_x3, 'y3': target_y3, 'z3': target_z3, 'tolerance': target_tolerance}
    if not combine:
        vector_attributes = {'name': vector_name, 'featureId': vector_featureId,
                             'pointsToLigand': vector_pointsToLigand,
                             'hasSyntheticProjectedPoint': vector_hasSyntheticProjectedPoint,
                             'optional': vector_optional,
                             'disabled': vector_disabled, 'weight': vector_weight}
        vector = et.SubElement(pharmacophore, 'vector', attrib=vector_attributes)
        et.SubElement(vector, 'origin', attrib=origin_attributes)
        et.SubElement(vector, 'target', attrib=target_attributes)
    else:
        featureCloud_attributes = {'name': vector_name, 'featureColor': feature[-3], 'featureId': vector_featureId,
                                  'optional': vector_optional, 'disabled': vector_disabled, 'weight': vector_weight,
                                  'orig_superfeature': feature[-1], 'envPatner': feature[-2]}
        featureCloud = et.SubElement(pharmacophore, 'featureCloud', attrib = featureCloud_attributes)
        et.SubElement(featureCloud, 'position', attrib=origin_attributes)
        for coord in points_new_sp:
            additionalPoint = et.SubElement(featureCloud, "additionalPoint", 
                                            x3=f"{coord[0]}", y3=f"{coord[1]}", z3=f"{coord[2]}", 
                                            weight="1.0")
    
    return

def pml_feature(pharmacophore, feature, points_new_sp, combine = True):
    """ This function distributes features according to their feature type to the appropriate feature function. 
        Code refered to Pyrod https://github.com/wolberlab/pyrod
    Input:
        pharmacophore: ElementTree Node - parent node in xml tree
        feature: list - contains feature information for outputting
        points_new_sp: ndarray - contains coordinates of all points within this new superfeature
        combine: bool - True for writing condense pharmacophore and point together
                        False for only writing condense pharmacophore
    """
    if feature[1] in ['H', 'NI', 'PI']:
        pml_feature_point(pharmacophore, feature, points_new_sp, combine = combine)
    elif feature[1] in ['HBA', 'HBD']:
        pml_feature_vector(pharmacophore, feature, points_new_sp, combine = combine)
    elif feature[1] == 'AR':
        pml_feature_plane(pharmacophore, feature, points_new_sp, combine = combine)
    return


def indent_xml(element, level=0):
    """ This function adds indentation to an xml structure for pretty printing. 
        Code from Pyrod https://github.com/wolberlab/pyrod"""
    i = "\n" + level*"  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = i + "  "
        if not element.tail or not element.tail.strip():
            element.tail = i
        for element in element:
            indent_xml(element, level + 1)
        if not element.tail or not element.tail.strip():
            element.tail = i
    else:
        if level and (not element.tail or not element.tail.strip()):
            element.tail = i


def pml_pharmacophore(features, points_new_sp, output_directory, name, combine = True):
    """ This function generates an xml tree describing a pharmacophore that is written to a pml file.
        Code refered to Pyrod https://github.com/wolberlab/pyrod
    Input:
        features: list - superfeature extracted for outputting from extract_superfeature()
        points_new_sp: list - list of ndarray containing coordinates of all points within each superfeature
        output_directory: str - Path to output directory
        name: str - name of output file
        combine: bool - True for writing condense pharmacophore and point together
                        False for only writing condense pharmacophore
    """
    pharmacophore = et.Element('pharmacophore', attrib={'name': name, 'pharmacophoreType': 'LIGAND_SCOUT'})
    for idx, feature in enumerate(features):
        pml_feature(pharmacophore, feature, points_new_sp[idx], combine = combine)
    indent_xml(pharmacophore)
    tree = et.ElementTree(pharmacophore)
    name = '{}.{}'.format(f"{name}", 'pml')
    tree.write('{}/{}'.format(output_directory, name), encoding="UTF-8", xml_declaration=True)
    return


def write_pharmacophore(data, model, output_directory, include_time = False, frequency_cutoff = None, 
                        noise_cutoff = 0.3, name = "cluster_pharmacophore", combine = True):
    """ This function writes out pharmacophores with or without points cloud
       Inputs:
           data: dict
           model: KMedoids model object -  e.g. KMedoids(method='pam', metric='manhattan', n_clusters=3)
           output_directory: str - Path to output directory
       Attributes:
           include_time: bool - True for including time distance
           frequency_cutoff: dict - used in extract_superfeature() frequency_cutoff for different superfeatures for dropping noise superfeature in certain cluster
           noise_cutoff: float - used in extract_superfeature() for dropping less dominent clusters within each superfeature
           name: str - name of output file
           combine: bool - True for writing condense pharmacophore and point together
                           False for only writing condense pharmacophore
    """
    wrap_data = parsers.get_wrap_data(data, model)
    if combine:
        name = name + "_points"
    for cluster_id in range(np.max(model.labels_)+1):
        features, points_new_sp = extract_superfeature(data, wrap_data, pose_nr = cluster_id, include_time = False, 
                                    frequency_cutoff = frequency_cutoff, noise_cutoff = float(noise_cutoff))
        pml_pharmacophore(features, points_new_sp, output_directory, f"{name}_{cluster_id}", combine = combine)
        if combine == True:
            print(f"Pharmacophore and associated points of cluster {cluster_id} written to {output_directory}")
        else:
            print(f"Pharmacophore of cluster {cluster_id} written to {output_directory}")
    return


def write_points(data, model, output_directory):
    '''This function writes out pharmacophores points in each cluster
       Inputs:
           data: dict
           model: KMedoids model object -  e.g. KMedoids(method='pam', metric='manhattan', n_clusters=3)
           output_directory: str - Path to output directory
    '''
    wrap_data = parsers.get_wrap_data(data,model)
    keys_ = list(data.keys())
    cluster_ids = list(np.unique(model.labels_))
    
    for cluster_id in cluster_ids:
        pharmacophore = et.Element("pharmacophore", name = "dynophore_dyno", id="pharmacophore", pharmacophoreType="LIGAND_SCOUT")
        for i in range(len(wrap_data)):
            data_ = wrap_data[i] # data for one superfeature
            data_ = data_[data_[:, -1] == cluster_id]  # data for cluster 0 in one superfeature
            if len(data_):
                name_ = parsers.get_feature_name(keys_[i])  # return "H"
                featureColor_ = data[keys_[i]]["color"]
                id_ = f"feature{i}"
                featureCloud = et.SubElement(pharmacophore, "featureCloud", name=name_, 
                                             featureColor=featureColor_, optional="false", 
                                             disabled="false", weight="1.0", id=id_)

                center_x, center_y, center_z = compute.get_geo_center(data_)
                position = et.SubElement(featureCloud, "position", x3=f"{center_x}", y3=f"{center_y}", z3=f"{center_z}")
                for j in range(len(data_)):
                    additionalPoint = et.SubElement(featureCloud, "additionalPoint", 
                                                    x3=f"{data_[j][0]}", y3=f"{data_[j][1]}", z3=f"{data_[j][2]}", 
                                                    weight="1.0")
        indent_xml(pharmacophore)
        tree = et.ElementTree(pharmacophore)
        tree.write(f"{output_directory}points_cluster_{cluster_id}.pml")
        print(f"Pharmacophore points of cluster {cluster_id} written to {output_directory}")
        
        
