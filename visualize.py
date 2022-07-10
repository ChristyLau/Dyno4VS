#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats.kde import gaussian_kde
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

def plot_heatmap(data, dim = 0, figsize = (7,4)):
    '''Function to plot density distribution
       Input:
           data: ndarray
       Attribute:
           dim: int - 0 for plotting xy axis, 1 for yz
           figsize: tuple
       refered to https://stackoverflow.com/questions/36957149/density-map-heatmaps-in-matplotlib
    '''
    x, y = data[:, dim], data[:, dim+1]
    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    fig = plt.figure(figsize = figsize)
    ax1 = fig.add_subplot(111)
    im1 = ax1.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5)
    
    ax1.set_xlim(x.min(), x.max())
    ax1.set_ylim(y.min(), y.max())
    ax1.set_title("Density Heatmap")
    fig.colorbar(im1,orientation='vertical')
    plt.show()
    
def plot_histogram(distances, i = None, key = None):
    '''Function used in parameter prediction programme for plotting distance distribution for one feature points cloud
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
    plt.ylabel("Binding Mode")

    n_cluster = len(np.unique(model.labels_))
    counter = dict(collections.Counter(model.labels_))
    print("There are", n_cluster, "clusters")
    print(f"Frames count within each binding mode: {counter}")
#     plt.show()
    
    if savefig == True:
        plt.savefig(f"{output_directory}bar_code.png", dpi = 200, bbox_inches = "tight")


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
    max_frame = compute.get_max_frame(data)
    # split polar coordinates
    angles = np.linspace(0, 2*np.pi, data_length, endpoint=False)
    labels = list(data.keys())
    keys = list(data.keys())
    pi = math.pi
    
    feature = []
    for i in feature_per_state:
        feature_temp = feature_per_state[i]
        frame_nr = list(model.labels_).count(i)
        print("frame_nr:", frame_nr)
        temp = [(i*max_frame)/frame_nr for i in feature_temp.values()]
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
        ax.plot(angles, feature_temp, color = colors[i], label = f"mode {i}")
    ax.set_theta_zero_location('N')
    ax.set_rlim(xmax, xmin)
    ax.set_rlabel_position(270)
    ax.set_title("Interaction Frequency of Binding Modes", fontsize = 13)
    plt.legend(bbox_to_anchor = (1.55, 0.05))
#     plt.show()
    if savefig == True:
        plt.savefig(f"{output_directory}radar.png", dpi = 200, bbox_inches = "tight")
        
        
def plot_2d(data, model = None, key = None, figsize = (10, 3.5)):
    '''2D drawing of original/clustered data of all/selected superfeatures in xy and yz projection.
       Input:
           data: dict
       Attributes:
           model: KMedoids model object -  e.g. KMedoids(method='pam', metric='manhattan', n_clusters=3)
           key: str - superfeature name to observe cluster result
           figsize: tuple
    '''
    ax_props = {
        "xlabel": None,
        "ylabel": None,
    }
    if key != None:
        fig, Ax = plt.subplots(1, 2,
                           figsize=figsize)
        if model != None:
            for axi, d in enumerate([0, 1], 0):
                data[key]["clustering"].evaluate(
                    ax=Ax[axi], dim=(d, d+1),
                    ax_props=ax_props
                )
        else: # model == None, plot original data
            for axi, d in enumerate([0, 1], 0):
                data[key]["clustering"].evaluate(
                    ax=Ax[axi], dim=(d, d+1),
                    ax_props=ax_props,
                    original=True
                )
        Ax[0].annotate(f"{key}", (0.05, 0.95), xycoords="axes fraction", fontsize=10)
        
    else:
        color_dict = {}
        labels = {0: "x", 1: "y", 2: "z"}
        for key, data_ in data.items():
            color = data_['color']
            color_dict[key] = color
        superfeatures_colors = {superfeature_id: tuple(colors.hex2color(f"#{color}")) for superfeature_id, color in color_dict.items()}
        superfeature_name = list(data.keys())
        fig, axs = plt.subplots(1, 2, figsize = figsize)

        if model != None:
            wrap_data = parsers.get_wrap_data(data, model)
            for d in [0, 1]:
                for i in range(len(wrap_data)):
                    superfeature_data = wrap_data[i]
                    cluster_data, noise = superfeature_data[superfeature_data[:, -2] != 0], superfeature_data[superfeature_data[:, -2] == 0]
                    color = [list(superfeatures_colors.values())[i]] * len(cluster_data)
                    axs[d].scatter(cluster_data[:, d], cluster_data[:, d+1], 
                                      c = color, label = superfeature_name[i], s = 1)
                    axs[d].scatter(noise[:, d], noise[:, d+1], c = 'grey', s = 1)
                    axs[d].set(xlabel=f"${labels[d]}$", ylabel=f"${labels[d+1]}$")
            plt.suptitle("Clustering of all superfeatures")

        else: # model == None, plot original data
            for d in [0, 1]:
                for i, key in enumerate(data.keys()):
                    point_data = data[key]["non_norm"]
                    color = [list(superfeatures_colors.values())[i]] * len(point_data)
                    axs[d].scatter(point_data[:, d], point_data[:, d+1], 
                                      c = color, label = superfeature_name[i], s = 1)
                    axs[d].set(xlabel=f"${labels[d]}$", ylabel=f"${labels[d+1]}$")
                    plt.suptitle("Point cloud of all superfeatures")
        axs[d].legend(bbox_to_anchor=(1.1, 1.05))
        fig.tight_layout()
            

def plot_3d(data, model = None, key = None, figsize = (10, 10)):
    '''3D drawing of data after clustering of all superfeatures
       For interactive observation, please run %matplotlib notebook before call this function
       Input:
           data: dict
       Attributes:
           model: KMedoids model object -  e.g. KMedoids(method='pam', metric='manhattan', n_clusters=3)
           key: str - superfeature name to observe cluster result
           figsize: tuple
       '''
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111, projection='3d')
    if key != None:
        superfeature_data = data[key]["non_norm"]
        if model != None:
            colormap = {0: "grey", 1: "tab:blue", 2: "tab:orange", 3: "tab:green", 
                        4: "tab:red", 5: "tab:purple", 6: "tab:brown", 7: "tab:pink"}
            for state in np.unique(data[key]["clustering"].labels):
                superfeature_data_state = superfeature_data[np.where(data[key]["clustering"].labels==state)]
                ax.scatter(superfeature_data_state[:, 0], superfeature_data_state[:, 1], 
                           superfeature_data_state[:, 2], c = colormap[state], label = state, s = 2)
            ax.set_title(f"Clustering of {key}")
            ax.legend(bbox_to_anchor=(1.1, 1.05))
        else: # model == None, plot original data
            color  = data[key]["color"]
            color = colors.hex2color(f"#{color}")
            ax.scatter(superfeature_data[:, 0], superfeature_data[:, 1], superfeature_data[:, 2],
                           c = color, label = key, s = 2)
            ax.set_title(f"Point cloud of {key}")
            ax.legend(bbox_to_anchor=(1.1, 1.05))
            
    else: # key == None, plot all superfeatures
        color_dict = {}
        for key, data_ in data.items():
            color = data_['color']
            color_dict[key] = color
        superfeatures_colors = {superfeature_id: tuple(colors.hex2color(f"#{color}")) for superfeature_id, color in color_dict.items()}
        superfeature_name = list(data.keys())
        if model != None:
            wrap_data = parsers.get_wrap_data(data, model)
            for i in range(len(data)):
                superfeature_data = wrap_data[i]
                cluster_data, noise = superfeature_data[superfeature_data[:, -2] != 0], superfeature_data[superfeature_data[:, -2] == 0]
                color = [list(superfeatures_colors.values())[i]] * len(cluster_data)
                ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], c = color, label = superfeature_name[i], s = 2)
                ax.scatter(noise[:, 0], noise[:, 1], noise[:, 2], c = 'grey', s = 2)
            ax.set_title("Clustering of all superfeatures")
        else:  # model == None, plot original data
            for i, key in enumerate(data.keys()):
                superfeature_data = data[key]["non_norm"]
                color = [list(superfeatures_colors.values())[i]] * len(superfeature_data)
                ax.scatter(superfeature_data[:, 0], superfeature_data[:, 1],
                           superfeature_data[:, 2], c = color, 
                           label = superfeature_name[i], s = 2)
            ax.set_title("Point cloud of all superfeatures")
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
        pic = ax.scatter(x, y, z, c = frames, cmap = "copper_r", s = s)
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
        ax1.scatter(x, y, c = frames, cmap = "copper_r")
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f"{key}")
        
        ax2 = fig.add_subplot(122)
        pic = ax2.scatter(y, z, c = frames, cmap = "copper_r")
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

    colors = {0: "w", 1: "cornflowerblue", 2: "limegreen", 3: "orange", 4: "cornsilk", 5: "grainsboro", 6: "lightcyan"}
    
    max_cluster_idx = 0
    for j in freq_map:
        max_cluster_idx = max(max_cluster_idx, len(freq_map[j][0]))

    fig, axes = plt.subplots(nr_plt, 1, sharex = True)
    fig.set_figheight(2*nr_plt)
    fig.set_figwidth(6)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0)

    for i in range(nr_plt):
        list(axes)[i].text(0.05, 0.9, f"binding mode {i}", fontsize= 13)
        list(axes)[i].set_ylim(ymax = 1)
        bottom = 0
        for cluster_idx in range(1, max_cluster_idx):
            if cluster_idx == 1:
                list(axes)[i].bar(keys, freq_map[i][:,cluster_idx], 
                                  width = 0.5, color = colors[cluster_idx],
                               label = f"static state {cluster_idx}"  )
            else: # cluster_idx >= 2
                bottom = bottom + freq_map[i][:, cluster_idx-1]
                list(axes)[i].bar(keys, freq_map[i][:,cluster_idx], 
                                  width = 0.5,bottom = bottom, 
                                  color = colors[cluster_idx], 
                               label = f"static state {cluster_idx}")
    list(axes)[nr_plt-1].set_xticklabels(keys, rotation = 45, fontsize = 7, ha='right')
    list(axes)[0].legend()
#     plt.show()
    if savefig == True:
        plt.savefig(f"{output_directory}stacked_bar.png", dpi = 200, bbox_inches = "tight")

        
def one_line_visualize(data, model = None, key = None, three_d = False, figsize = (7,7)):
    '''A funtion to show
           2D plot pharmacophore points cloud before/after clustering
           3D plot pharmacophore points cloud before/after clustering
        Input:
           data: dict
       Attribute:
           model: KMedoids model object -  e.g. KMedoids(method='pam', metric='manhattan', n_clusters=3)
           three_d: bool - True for viewing 3D plot
           figsize: tuple
           '''
    if not three_d:
        plot_2d(data, model, key, figsize = figsize)
    else:
        plot_3d(data, model, key, figsize = figsize)
    
    
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