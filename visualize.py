#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import matplotlib.pyplot as plt
from matplotlib import colors
import collections
import numpy as np
import pandas as pd
import math
import MDAnalysis as mda
import compute, parsers
# import nglview
# import mdtraj
# from MDAnalysis.analysis import align, rms, diffusionmap
# get_ipython().run_line_magic('matplotlib', 'inline')


def plot_histogram(distances, i = None, key = None):
    '''Funtion used in parameter prediction programme for plotting distance distribution for one feature points cloud
       Input: 
           distances: ndarray - shape: (n, n) distance matrix with pairwise distance of points
       Output:
           y: ndarray - count of each bin (or interval)
       Attributes:
           i: int - idx of superfeature for creating figure title
           key: str - name of superfeature for creating figure title
           '''
    plt.figure(figsize=(5,4))
    weights = np.zeros_like(distances.flatten()) + 1. / distances.flatten().size
    y, x, _ = plt.hist(distances.flatten(), bins=100, color='y', weights=weights)
    plt.ylabel("frequency")
    plt.xlabel("pairwise distance")
    plt.title(f"{i} {key}")
    plt.axvline(min_value)
#     plt.show()
    
    return y


def plot_bar_code(model, savefig = False, output_directory = None, n_drop = 0):
    '''Funtion to show distribution of frames for different clusters
       Input:
           model: KMedoids model object -  e.g. KMedoids(method='pam', metric='manhattan', n_clusters=3)
       Attributes:
           savefig: bool - True for saving the figure to given path
           output_directory: str - path to output directory
           n_drop: int - Currently not in use. Developed for the function dropping first few frames.
    '''
    n_frames = len(model.labels_)
    x = np.array(range(n_frames)) + n_drop
    plt.figure(figsize=(10, 5), dpi=80)

    plt.scatter(x, model.labels_, marker = "|")
    plt.xlabel("Frame")
    plt.ylabel("Binding State")

    n_cluster = len(np.unique(model.labels_))
    counter = dict(collections.Counter(model.labels_))
    print("There are", n_cluster, "clusters")
    print(f"Frames count within each binding state: {counter}")
#     plt.show()
    
    if savefig == True:
        plt.savefig(f"{output_directory}bar_code.png", dpi = 200, bbox_inches = "tight")

# def plot_radar(feature_per_state, xmin = 0, xmax = 1):
#     '''Plot interaction frequency within each cluster as radar plot'''
#     state_count = len(feature_per_state.keys())
#     data_length = len(feature_per_state[0])
#     # split polar coordinates
#     angles = np.linspace(0, 2*np.pi, data_length, endpoint=False)
#     labels = [key for key in feature_per_state[0].keys()]

#     feature = []
#     for i in feature_per_state:
#         feature_temp = feature_per_state[i]
#         temp = [i for i in feature_temp.values()]
#         feature.append(temp)

#     angles = np.concatenate((angles, [angles[0]]))
#     labels = np.concatenate((labels, [labels[0]]))
    
#     fig = plt.figure(figsize=(8, 6), dpi=100)
    
#     ax = plt.subplot(111, polar=True)
#     feature_map = {
#         k: np.concatenate((feature[k], [feature[k][0]]))
#         for k in range(state_count)
#     }

#     colors = ["g", "b", "r", "y", "m", "k", "c"]
#     for i in feature_map:
#         feature_temp = feature_map[i]
#         ax.plot(angles, feature_temp, color = colors[i], label = f"state {i}")

#     ax.set_thetagrids(angles*180/np.pi, labels)
#     ax.set_theta_zero_location('N')
#     ax.set_rlim(xmax, xmin)
#     ax.set_rlabel_position(270)
#     ax.set_title("interaction frequency of cluters")
#     plt.legend(bbox_to_anchor = (1.2, 1.05))
#     plt.show()

def plot_radar(data, model, xmin = 0, xmax = 1, savefig = False, output_directory = None):
    '''Funtion to show frequency of each superfeature among different clusters in a radar form
       Input:
           data: dict
           model: KMedoids model object -  e.g. KMedoids(method='pam', metric='manhattan', n_clusters=3)
       Attributes:
           xmin: float - minimal xtick
           xmax: float - maximal xtick
           savefig: bool - True for saving the figure to given path
           output_directory: str - path to output directory
    '''
    feature_per_state = compute.get_feature_freq_per_state(data, model)
    state_count = len(feature_per_state.keys())
    data_length = len(feature_per_state[0])
    # split polar coordinates
    angles = np.linspace(0, 2*np.pi, data_length, endpoint=False)
    labels = list(data.keys())
    keys = list(data.keys())
    pi = math.pi
    
    feature = []
    for i in feature_per_state:
        feature_temp = feature_per_state[i]
        temp = [i for i in feature_temp.values()]
        feature.append(temp)
    # Generate radar board
    fig=plt.figure(figsize=(10,5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles, keys)
    for label,i in zip(ax.get_xticklabels(),range(0,len(angles))):
        angle_rad=angles[i]
        if angle_rad <= pi/2:
            ha= 'left'
            va= "bottom"
            angle_text=angle_rad*(-180/pi)+90
        elif pi/2 < angle_rad <= pi:
            ha= 'left'
            va= "top"
            angle_text=angle_rad*(-180/pi)+90
        elif pi < angle_rad <= (3*pi/2):
            ha= 'right'
            va= "top"  
            angle_text=angle_rad*(-180/pi)-90
        else:
            ha= 'right'
            va= "bottom"
            angle_text=angle_rad*(-180/pi)-90
        label.set_rotation(angle_text)
        label.set_verticalalignment(va)
        label.set_horizontalalignment(ha)

    # Plot  
    angles = np.concatenate((angles, [angles[0]]))
    feature_map = {
        k: np.concatenate((feature[k], [feature[k][0]]))
        for k in range(state_count)
    }
    colors = ["g", "b", "r", "y", "m", "k", "c"]
    for i in feature_map:
        feature_temp = feature_map[i]
        ax.plot(angles, feature_temp, color = colors[i], label = f"state {i}")
    ax.set_theta_zero_location('N')
    ax.set_rlim(xmax, xmin)
    ax.set_rlabel_position(270)
    ax.set_title("Interaction frequency of cluters", fontsize = 20)
    plt.legend(bbox_to_anchor = (1.55, 0.05))
#     plt.show()
    if savefig == True:
        plt.savefig(f"{output_directory}radar.png", dpi = 200, bbox_inches = "tight")

def plot_2d_wrap_data(data, model, d = 0,  figsize = (7, 7)):
    '''2D drawing of data after clustering of all superfeatures
       Input:
           data: dict
           model: KMedoids model object -  e.g. KMedoids(method='pam', metric='manhattan', n_clusters=3)
       Attribute:
           d: int - first dimension to start. 0 for without time clustering, 0 or 1 for with time.
           figsize: tuple
    '''
    wrap_data = parsers.get_wrap_data(data, model)
        
    color_dict = {}
    for key, data_ in data.items():
        color = data_['color']
        color_dict[key] = color
    superfeatures_colors = {superfeature_id: tuple(colors.hex2color(f"#{color}")) for superfeature_id, color in color_dict.items()}
    superfeature_name = list(data.keys())
    fig, ax = plt.subplots(figsize = figsize)
    for i in range(len(wrap_data)):
        superfeature_data = wrap_data[i]
        cluster_data, noise = superfeature_data[superfeature_data[:, 3] != 0], superfeature_data[superfeature_data[:, 3] == 0]
        color = [list(superfeatures_colors.values())[i]] * len(cluster_data)
        
        ax.scatter(cluster_data[:, d], cluster_data[:, d+1], c = color, label = superfeature_name[i], s = 1)
        ax.scatter(noise[:, d], noise[:, d+1], c = 'grey', s = 1)
    if d == 0:
        plt.xlabel("$x$")
        plt.ylabel("$y$")
    else:
        plt.xlabel("$y$")
        plt.ylabel("$z$")
    ax.set_title("Clustering of all superfeatures")
    ax.legend(bbox_to_anchor=(1.1, 1.05))

def plot_3d_wrap_data(data, model, figsize = (10, 10)):
#     get_ipython().run_line_magic('matplotlib', 'notebook')
    '''3D drawing of data after clustering of all superfeatures
       For interactive observation, please run %matplotlib notebook before call this function
       Input:
           data: dict
           model: KMedoids model object -  e.g. KMedoids(method='pam', metric='manhattan', n_clusters=3)
       Attribute:
           figsize: tuple
       '''
    wrap_data = parsers.get_wrap_data(data, model)
        
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111, projection='3d')

    color_dict = {}
    for key, data_ in data.items():
        color = data_['color']
        color_dict[key] = color
    superfeatures_colors = {superfeature_id: tuple(colors.hex2color(f"#{color}")) for superfeature_id, color in color_dict.items()}
    superfeature_name = list(data.keys())
    for i in range(len(data)):
        superfeature_data = wrap_data[i]
        cluster_data, noise = superfeature_data[superfeature_data[:, 3] != 0], superfeature_data[superfeature_data[:, 3] == 0]
        color = [list(superfeatures_colors.values())[i]] * len(cluster_data)

        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], c = color, label = superfeature_name[i], s = 1)
        ax.scatter(noise[:, 0], noise[:, 1], noise[:, 2], c = 'grey', s = 2)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    ax.set_title("Clustering of all superfeatures")
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    
        
def plot_feature_along_traj(data, key, norm = True, three_d = False, figsize = (3, 3), s = 2):
    '''A funtion to view time sequency of pharmacophore point cloud for 1 superfeature.
       The points appears earlier is colored lighter whereas the ones appears later are darker.
       Input:
           data: dict
           key: str - e.g. "H[3185,3179,3187,3183,3181,3178]"
       Attributes:
           norm: bool - True to view normed coordinates, otherwise the original coords
           three_d: bool - True for plotting 3D figure
           figsize: tuple - modify the figure size
           s: int - marker size
       '''
    if norm == False:
        x, y, z = data[key]["non_norm"][:, 0], data[key]["non_norm"][:, 1], data[key]["non_norm"][:, 2]
    else:
        x, y, z = data[key]["points"][:, 0], data[key]["points"][:, 1], data[key]["points"][:, 2]
    frames = data[key]["frames"]
    if three_d:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
        pic = ax.scatter(x, y, z, c = frames, cmap = "YlOrBr", s = s)
        plt.colorbar(pic)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(f"{key}")
    else: # 2D
        xlim = figsize[0] * 2
        ylim = figsize[0]
        fig = plt.figure(figsize=(xlim, ylim))
        ax1 = fig.add_subplot(121)
        ax1.scatter(x, y, c = frames, cmap = "YlOrBr")
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f"{key}")
        
        ax2 = fig.add_subplot(122)
        pic = ax2.scatter(y, z, c = frames, cmap = "YlOrBr")
        ax2.set_xlabel('y')
        ax2.set_ylabel('z')
        fig.colorbar(pic)
#     plt.show()
    
    
def plot_stacked_bar(data, model, savefig = False, output_directory = None):
    '''Plot frequency of each cluster within each superfeatures for different binding pose
       Input:
           data: dict
           model: KMedoids model object -  e.g. KMedoids(method='pam', metric='manhattan', n_clusters=3)
       Attribute:
           savefig: bool - True for saving figure to given path
           output_directory: str - Path to output directory
       '''
    keys = data.keys()
    state_statistis = compute.get_state_statistis(data, model)
    max_frame = compute.get_max_frame(data)
    nr_plt = len(list(state_statistis.keys()))
    freq_map = {
#         k: ((pd.DataFrame.from_dict(state_statistis[k]).fillna(0))/max_frame).T.to_numpy()
        k: ((pd.DataFrame.from_dict(state_statistis[k]).fillna(0))/max_frame)
        for k in range(nr_plt)
    }
    # just for in case states have different number of maximal static clusters
    min_state_idx, max_state_idx = 99, 0
    for k in range(nr_plt):
        state_idx_tmp = freq_map[k].shape[0]
        min_state_idx = min(min_state_idx, state_idx_tmp)
        max_state_idx = max(max_state_idx, state_idx_tmp)
    for k in range(nr_plt):
        for i in range(min_state_idx, max_state_idx):
            exist = True
            try:
                try_ = freq_map[k].iloc[i]
            except:
                exist = False
            if not exist:
                freq_map[k].loc[i] = [0]*len(keys)
        freq_map[k] = freq_map[k].T.to_numpy()

    colors = {0: "w", 1: "cornflowerblue", 2: "limegreen", 3: "lightcyan", 4: "cornsilk", 5: "grainsboro", 6: "orange"}
    
    max_cluster_idx = 0
    for j in freq_map:
        max_cluster_idx = max(max_cluster_idx, len(freq_map[j][0]))

    fig, axes = plt.subplots(nr_plt, 1, sharex = True)
    fig.set_figheight(2*nr_plt)
    fig.set_figwidth(6)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0)

    for i in range(nr_plt):
        list(axes)[i].text(0.05, 0.9, f"state {i}", fontsize= 15)
        list(axes)[i].set_ylim(ymax = 1)
        bottom = 0
        for cluster_idx in range(1, max_cluster_idx):
            if cluster_idx == 1:
                list(axes)[i].bar(keys, freq_map[i][:,cluster_idx], 
                                  width = 0.5, color = colors[cluster_idx],
                               label = f"cluster {cluster_idx}"  )
            else: # cluster_idx >= 2
                bottom = bottom + freq_map[i][:, cluster_idx-1]
                list(axes)[i].bar(keys, freq_map[i][:,cluster_idx], 
                                  width = 0.5,bottom = bottom, 
                                  color = colors[cluster_idx], 
                               label = f"cluster {cluster_idx}")
    list(axes)[nr_plt-1].set_xticklabels(keys, rotation = 45, fontsize = 7, ha='right')
    list(axes)[0].legend()
#     plt.show()
    if savefig == True:
        plt.savefig(f"{output_directory}stacked_bar.png", dpi = 200, bbox_inches = "tight")

        
def one_line_visualize(data, model, three_d = False, figsize = (7,7)):
    '''A funtion to show
           2D plot pharmacophore points cloud after clustering
           3D plot pharmacophore points cloud after clustering
        Input:
           data: dict
           model: KMedoids model object -  e.g. KMedoids(method='pam', metric='manhattan', n_clusters=3)
       Attribute:
           three_d: bool - True for viewing 3D plot
           figsize: tuple
           '''
    if not three_d:
        for d in [0, 1]:
            plot_2d_wrap_data(data, model, d = d, figsize = figsize)
    else:
        plot_3d_wrap_data(data, model, figsize = figsize)
    
    
def one_line_analysis(data, model, xmin = 0, xmax = 0.8, savefig = False, output_directory = None):
    '''A condense function to 
           show frames for each cluster (plot_bar_code)
           show cluster distribution of each feature for different binding states
           show frequency of superfeatures in each cluster
           print the interaction summary for each cluster
        Input:
           data: dict
           model: KMedoids model object -  e.g. KMedoids(method='pam', metric='manhattan', n_clusters=3)
       Attributes:
           xmin: float - minimal xtick for radar plot
           xmax: float - maximal xtick for radar plot
           savefig: bool - True for saving figure to given path
           output_directory: str - Path to output directory
    '''
    plot_bar_code(model, savefig = savefig, output_directory = output_directory)
    plot_stacked_bar(data, model, savefig = savefig, output_directory = output_directory)
#     feature_per_state = compute.get_feature_freq_per_state(data, model)
    plot_radar(data, model, xmin = xmin, xmax = xmax, savefig = savefig, output_directory = output_directory)
    interact_summary = compute.get_interact_summary(data, model)
    print(interact_summary)

#################################### NOT IN USE ##############################################
def plot_whole_rmsd(pdb_path, dcd_path, select = 'chainID X', alignment = True, n_drop = 0):
    '''Plot RMSD of ligand-target complex within each cluster
       Currently not in use.
    '''
    u = mda.Universe(pdb_path, dcd_path, in_memory=True)
    ligand = u.select_atoms(select)
    
    rmsd_ls = []
    X =  np.arange(n_drop, len(u.trajectory))
    
    if alignment:
        reference_coordinates = u.trajectory.timeseries(asel = ligand).mean(axis = 1)
        reference = mda.Merge(ligand).load_new(
                    reference_coordinates[:, None, :], order = "afc")
        ref_ca = reference.select_atoms(select)
        aligner = align.AlignTraj(u, reference, select = select, in_memory = True).run()
    
        # get RMSD
        rmsd_ls = []
        X =  range(len(u.trajectory))
    
    for i in X:
        mobile = mda.Universe(pdb_path, dcd_path)
        mobile.trajectory[i]
        mobile_ca = mobile.select_atoms(select)
        rmsd_ = rms.rmsd(mobile_ca.positions, ligand.positions, superposition=False)
        rmsd_ls.append(rmsd_)
        
    plt.figure(figsize=(15,5))
    plt.plot(X, rmsd_ls)
    plt.ylim(bottom = 0)
    plt.title(f"RMSD of ligand: {round(np.mean(np.array(rmsd_ls)), 2)}")
    plt.show()

    
def plot_cluster_rmsd(model, pdb_path, cluster_dcd_path, select = 'chainID X'):
    '''Plot RMSD of ligand within each cluster
       Currently not in use.
       TODO: RMSD of ligand and involving envPartner should be computed
    '''
    n_cluster = np.max(model.labels_) + 10
    numRows =  math.ceil(n_cluster/3)
    numCols = 3
    
    plt.figure(figsize=(20,15))
    for k in np.unique(model.labels_):
        u = mda.Universe(pdb_path, f"{cluster_dcd_path}cluster_{k}.dcd", in_memory = True)
        ligand = u.select_atoms(select)
        # reference = average structure
        reference_coordinates = u.trajectory.timeseries(asel = ligand).mean(axis = 1)
        reference = mda.Merge(ligand).load_new(
                    reference_coordinates[:, None, :], order = "afc")
        ref_ca = reference.select_atoms(select)
        aligner = align.AlignTraj(u, reference, select = select, in_memory = True).run()

        rmsd_ls = []
        X =  range(len(u.trajectory))
        mobile = mda.Universe(pdb_path, f"{dcd_path}cluster_{k}.dcd")

        for j in X:
            mobile.trajectory[j]
            mobile_ca = mobile.select_atoms(select)

            rmsd_ = rms.rmsd(mobile_ca.positions, ref_ca.positions, superposition = False)
            rmsd_ls.append(rmsd_)

        # plot
        plt.subplot(numRows, numCols, (k + 1))
        max_rmsd = np.max(np.array(rmsd_ls)) + 0.5
        plt.ylim(0, max_rmsd)
        plt.plot(X, rmsd_ls)
        plt.title(f"State {k} avg. RMSD = {round(np.mean(np.array(rmsd_ls)), 2)}")
        plt.xlabel("Frames")
        plt.ylabel("RMSD ($\AA$)")
    plt.tight_layout() 
    
def plot_diffusionmap(pdb_path, dcd_path, select = "protein or chainID X", reduce = True):
    '''Plot pairwise frame diffusionmap
       Currently not in use.
    '''
    if reduce:
        compute.reduce_frames(pdb_path, dcd_path, select = select, out_path = f"output/reduced.dcd", final_n_frame = 500)
        dcd_path = f"output/reduced.dcd"
    u = mda.Universe(pdb_path, dcd_path)
    aligner = align.AlignTraj(u, u, select = select,
                             in_memory = True).run()
    
    diffusion_matrix = diffusionmap.DistanceMatrix(u, select = 'chainID X').run()
    plt.imshow(diffusion_matrix.dist_matrix, cmap='viridis')
    plt.xlabel('Frame')
    plt.ylabel('Frame')
    plt.colorbar(label=r'RMSD ($\AA$)')
    plt.show()
    
def view_trajectory(model, pdb_path, dcd_path, state = None):
    '''view trajectory
       Currently not in use, as nglview requires direct call from jupyter notebook'''
    cluster_frames_map = {k: np.where(model.labels_ == k)[0] for k in range(4)}
    trajectory = mdtraj.load_dcd(dcd_path, top = pdb_path)
    cluster_traj_map = {k: trajectory[cluster_frames_map[k]] for k in range(4)}
    if state == None:
        states = [x for x in np.unique(model.labels_)]
    for state in states:
        print(f"State {state}")
        nglview.show_mdtraj(cluster_traj_map[state])