#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tqdm import tqdm
import numpy as np
from cnnclustering import cluster
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn_extra import cluster as skecluster
import visualize, compute, parsers, write

def do_cluster(data, include_time = False, redo = None, plot = True, info_table = True):
    '''A function to cluster data within each superfeature sequently
       Input:
           data: dict
       Attributes:
           include_time: bool - True for including time distance
           redo: list - a list of feature names to recluster
           plot: bool - True for plotting cluster result
           info_table: bool - True for print parameter chosen and predict cluster features of each superfeature
       Outputs:
           (data, need_redo): tuple
               data: dict - data with fitted cluster
               need_redo: dict - superfeatures needed redo
                                 e.g:
                                 - superfature name
                                   - reasons
                                   - superfeature index
               '''
    # setup visualization params
    mpl.rcParams["figure.dpi"] = 120
    ax_props = {
        "xlabel": None,
        "ylabel": None,
    }
    if not include_time:
        default_cluster_params = {
            "radius_cutoff": 0.2,
            "cnn_cutoff": 5,
            "member_cutoff": 10
        }
    else:
        default_cluster_params = {
            "radius_cutoff": 0.2,
            "cnn_cutoff": 20,
            "member_cutoff": 10
        }
    need_redo = {}
    
    if not redo:
        redo = data.keys()
        
    for i, (fkey, data_) in enumerate(data.items()):
        if "clustering" not in data[fkey]:
            if include_time:
                data[fkey]["clustering_"] = cluster.Clustering(data_["distances"], registered_recipe_key="distances")
                data[fkey]["clustering"] = cluster.Clustering(data_["points"])     
            else:
                data[fkey]["clustering"] = cluster.Clustering(data_["points"])
        
        if fkey in redo:
            print(i, fkey)
            if include_time:
                data[fkey]["clustering_"].fit(**data[fkey].get("params", default_cluster_params), v = info_table)
                data[fkey]["clustering"].labels = data[fkey]["clustering_"].labels
                data[fkey]["clustering"]._summary = data[fkey]["clustering_"]._summary
            else:
                data[fkey]["clustering"].fit(**data[fkey].get("params", default_cluster_params), v = info_table)

            if plot:
                fig, Ax = plt.subplots(1, 4,
                           figsize=(mpl.rcParams["figure.figsize"][0] * 3, mpl.rcParams["figure.figsize"][1] * 1.2))
                for axi, d in enumerate([0, 1]):
                    data[fkey]["clustering"].evaluate(
                        ax=Ax[axi], dim=(d, d+1),
                        original=True,
                        ax_props=ax_props
                    )
                for axi, d in enumerate([0, 1], 2):
                    data[fkey]["clustering"].evaluate(
                        ax=Ax[axi], dim=(d, d+1),
                        ax_props=ax_props
                    )
                Ax[0].annotate(f"{i}: {fkey}", (0.05, 0.95), xycoords="axes fraction", fontsize=10)
                plt.show()
            
            n_cluster_real = list(data[fkey]["clustering"]._summary.to_DataFrame()["n_clusters"])[-1]
            ratio_noise = list(data[fkey]["clustering"]._summary.to_DataFrame()["ratio_noise"])[-1]

            reasons = []
            if n_cluster_real < data[fkey]['min_cluster_n'] or ratio_noise == 1:
                print(f"Discrepancy in predicted cluster number! Now: {n_cluster_real}  Expected: {data[fkey]['min_cluster_n']}")
                reasons.append("n_clusters")
            if ratio_noise > 0.05 and ratio_noise < 1:
                print("Noise over 5%!")
                reasons.append("noise")
            if len(reasons):
                need_redo[fkey] = {"reasons": reasons, "idx": i}
    if need_redo:
        print("Following features will be reclustered")
        
        # formating info table
        print("{:<4} {:<35} {:<15}".format('idx', 'key', 'reason'))
        max_len_key = 0
        for key_ in need_redo.keys():
            max_len_key = max(max_len_key, len(key_))
        for key_, value_ in need_redo.items():
            idx, reason = value_["idx"], value_["reasons"]
            print("{:<4} {:<35} {:<15}".format(idx, key_, str(reason)))
    else:
        print("From computer's view, no cluster need manual parameter adjustment")
    return data, need_redo


def parameter_scan(data, key, 
                   r_start = 0.05, r_end = 0.3, r_step = 0.05, 
                   c_start = 0, c_end = 50, c_step = 2, 
                   include_time = False, min_cluster_n = None,
                   plot = False, info_table = False):
    '''Parameter scan for data of specific superfeature. If suitable params found, replace automatically.
       Input:
           data: dict
           key: str
       Attributes:
           r_start: float - start search value for radius_cutoff
           r_end: float - end search value for radius_cutoff
           r_step: float - step in increasing radius_cutoff
           c_start: int - start search value for cnn_cutoff
           c_end: int - end search value for cnn_cutoff
           c_step: int - step in increasing cnn_cutoff
           include_time: bool - True for including time distance
           min_cluster_n: int - wanted cluster number
           plot: bool - True for plotting cluster results after parameter each fine-tuning
           info_table: bool - True for printing plotting parameter table after each fine-tuning
       Output:
           (radius_cutoff, cnn_cutoff, member_cutoff) : tuple
               new parameters if found, otherwise old parameters
               - radius_cutoff: radius within which common neighbours to be searched
               - cnn_cutoff: the number of common nearest neighbours two points need to share with respect to the radius to be assigned to the same cluster (i.e. the similarity threshold).
               - member_cutoff: the least points within one cluster
    '''
    try:
        if include_time:
            cluster_ = data[key]["clustering_"]
        else:
            cluster_ = data[key]["clustering"]
    except:
        if include_time:
            data[key]["clustering_"] = cluster.Clustering(data[key]["distances"], registered_recipe_key="distances")
            cluster_ = data[key]["clustering_"]
        else:
            data[key]["clustering"] = cluster.Clustering(data[key]["points"])
            cluster_ = data[key]["clustering"]
    if not min_cluster_n:
        min_cluster_n = data[key]["min_cluster_n"]
    orig_radius_cutoff, orig_cnn_cutoff, orig_member_cutoff = data[key]["params"]["radius_cutoff"], data[key]["params"]["cnn_cutoff"], data[key]["params"]["member_cutoff"]
    for r in tqdm(np.arange(r_start, r_end, r_step)):
        for c in np.arange(c_start, c_end, c_step):
            # fit from pre-calculated distances
            cluster_.fit(r, c, member_cutoff=10, v=False)
#             data[key]["clustering"].fit(r, c, member_cutoff=10, v=False)
    
    # Get summary sorted by number of identified clusters
    df = cluster_.summary.to_DataFrame()
#     df = data[key]["clustering"].summary.to_DataFrame()
    df = df[(df.n_clusters == min_cluster_n)]
    df = df[(df.ratio_noise > 0)].sort_values(["ratio_noise", "radius_cutoff"])
    try:
        radius_cutoff, cnn_cutoff, ratio_noise = df.iloc[0][["radius_cutoff", "cnn_cutoff", "ratio_noise"]]
    except:
        print(f"Cannot find solution for {key} given required cluster number and noise ratio")
        return (orig_radius_cutoff, orig_cnn_cutoff, orig_member_cutoff)
    
    # save the corrected parameters and cluster object
    if ratio_noise < 0.05:
        data[key]["params"]["radius_cutoff"], data[key]["params"]["cnn_cutoff"] = radius_cutoff, cnn_cutoff
        data, need_redo = do_cluster(data, include_time = include_time, 
                                       plot = plot, redo = [key], info_table = info_table)
#         cluster_.fit(radius_cutoff, cnn_cutoff, member_cutoff=10, v=True)
#         data[key]["clustering"].fit(radius_cutoff, cnn_cutoff, member_cutoff=10, v=True)
        print("Solution found with parameter scan")
        return (radius_cutoff, cnn_cutoff, orig_member_cutoff)
    else:
        print(f"Cannot find solution for {key} given cluster number and noise ratio")
        return (orig_radius_cutoff, orig_cnn_cutoff, orig_member_cutoff)
    

def params_adjust(data, need_redo, include_time = False, repeat = 10, plot = False, info_table = False):
    '''Adjust parameter for superfeatures in need_redo
       Input:
           data: dict
           need_redo: dict - superfeatures needed redo
                             e.g:
                             - superfature name
                               - reasons
                               - superfeature index
       Attributes:
           include_time: bool - True for including time distance
           repeat: int - repeat time in the first method used in params_adjust() programm
           plot: bool - True for plotting cluster results after parameter each fine-tuning
           info_table: bool - True for printing plotting parameter table after each fine-tuning
       Output:
           data: dict
           please_manual: list - a list of feature names which suggested to define parameters manually
    '''
    please_manual = []
    need_redo_copy = need_redo.copy()
    print("Start parameter adjustment programme")
    for key in list(need_redo_copy.keys()):
        print("*"*100)
        print(f"Start working on {key}")
        
        info = need_redo_copy[key]
        fix_status = False
        reasons = info["reasons"]
        orig_radius_cutoff, orig_cnn_cutoff, orig_member_cutoff = data[key]["params"].values()
        cnn_cutoff = orig_cnn_cutoff
        # only the noise problem
        if reasons == ["noise"]:
            time = 0
            print("Start adjusting cnn_cutoff")
            while time < repeat and cnn_cutoff >= 3 and fix_status == False:  # if number of clusters wrong or noise is decreased, no need to continue running
                time += 1
                cnn_cutoff -= 2
                data[key]["params"]['cnn_cutoff'] = cnn_cutoff
                data_temp, need_redo_temp = do_cluster(data, include_time = include_time, 
                                                       plot = plot, redo = [key], info_table = info_table)
                if key not in need_redo_temp.keys():
                    fix_status = True
                    new_radius_cutoff, new_cnn_cutoff, new_member_cutoff = (orig_radius_cutoff, cnn_cutoff, orig_member_cutoff)
            if not fix_status:
                (new_radius_cutoff, new_cnn_cutoff, new_member_cutoff) = parameter_scan(data, key, include_time = include_time,
                                                                                       plot = plot, info_table = info_table)

        # if involve the problem of unsatisfied cluster number
        else:
            (new_radius_cutoff, new_cnn_cutoff, new_member_cutoff) = parameter_scan(data, key, include_time = include_time,
                                                                                   plot = plot, info_table = info_table)
            
        if (new_radius_cutoff, new_cnn_cutoff, new_member_cutoff) != (orig_radius_cutoff, orig_cnn_cutoff, orig_member_cutoff):
            fix_status = True
            
        if fix_status:
            del need_redo[key]
            data[key]["params"] = {
                "radius_cutoff": new_radius_cutoff,
                "cnn_cutoff": new_cnn_cutoff,
                "member_cutoff": new_member_cutoff
            }
            print(f"Solution found for {key}")
        else:
            please_manual.append(key)
            print(f"Failed for {key}")
    return data, please_manual


def get_binding_pose_cluster_inertia(one_hot_matrix, min_cluster = 2, max_cluster = 7):
    '''Scan cluster number for input one_hot_matrix
       plot inertia, i.e. the difference within each cluster given cluster number
       Input: 
           one_hot_matrix: ndarray - one-hot-key encoding matrix
                           e.g:
                           frame | existance of state 0 in superfeature 1 | existance of state 1 in superfeature 1...
                           1     | 1 (means exist)                        | 0 (means absence)
                           2     ...
                           3     ...
       Attributes:
           min_cluster: int - minimal cluster number
           max_cluster: int - maximal cluster number
    '''
    inertia = []
    for n_clusters in tqdm(range(min_cluster, max_cluster)):
        model = skecluster.KMedoids(n_clusters = n_clusters, metric = "manhattan", method = "pam")
        model.fit(one_hot_matrix)
        inertia.append(model.inertia_)
    # plot  
    fig, ax = plt.subplots()
    ax.plot(range(min_cluster, max_cluster), inertia)
    ax.set_xlabel("#clusters")
    ax.set_ylabel("inertia: difference within each cluster")
    plt.show()


def get_states_per_interaction(state_matrix):
    '''A funtion to get the count of clusters in static clustering for each superfeature
       Input:
           state_matrix: ndarray - state information of all superfeatures. Shape (max_frame_nr, superfature_count)
       Output:
           states_per_interaction: list - shape (superfature_count, n)
    '''
    return [max(x) + 1 for x in state_matrix.T]


def get_center_features(data, model, state_matrix = []):
    '''Get prominent features for Kemoids cluster centers
       Input: 
           data: dict
           model: KMedoids model object -  e.g. KMedoids(method='pam', metric='manhattan', n_clusters=3)
           state_matrix: ndarray (optional) - state information of all superfeatures. Shape (max_frame_nr, superfature_count)
       Output: 
            center_features: dict
                             - <binding state idx>: int
                               - superfeature name: str
                                 - state: cluster idx in static clustering
                                 - idx: superfeature idx
    '''
    if state_matrix == []:
        state_matrix = compute.get_state_matrix(data)
    center_features = {}
    features = list(data.keys())
    for i in range(model.cluster_centers_.shape[0]):
        print(i, ":")
        features_tmp = {}
        present_feature_pos = np.where(model.cluster_centers_[i] == 1)[0]
        states_per_interaction = get_states_per_interaction(state_matrix)

        for pos in present_feature_pos:
            feature_index, state = get_feature_state_from_onehot_position(pos, states_per_interaction)
            features_tmp[features[feature_index]] = {"state": state, "idx": feature_index}
            print(f"    {features[feature_index]:>40} state {state:<10}")
        
        center_features[i] = features_tmp
    return center_features

    
def get_feature_state_from_onehot_position(pos, states_per_interaction):
    '''A funtion to retrace which features and state is represented by a onehot matrix position
       Inputs:
           pos: int - position of certain feature in one-hot-key matrix
           states_per_interaction: list - shape (superfature_count, n)
       Output:
           (i, pos): tuple
               i : superfeature idx
               pos: cluster idx in static clustering
    '''
    cumsum = 0
    newcumsum = 0
    for i, state_count in enumerate(states_per_interaction):
        newcumsum += state_count
        if pos < newcumsum:
            return (i, pos - cumsum)
        cumsum = newcumsum
    return (i, pos)


def binding_state_cluster(data, n_clusters = None, print_center_features = False):
    '''A function to cluster different binding poses
       Input:
           data: dict
       Attributes:
           n_clusters: int - number of clusters wanted
           print_center_features: bool - True for printing prominent features for Kemoids cluster centers
       Output: 
           model: KMedoids object
    '''
    state_matrix = compute.get_state_matrix(data)
    one_hot_matrix = compute.get_adjusted_one_hot_encoding(data, compute.get_one_hot_encoding(state_matrix))
    
    if n_clusters == None:
        get_binding_pose_cluster_inertia(one_hot_matrix)
        n_clusters = int(input("Please give cluster number: "))
    model = skecluster.KMedoids(n_clusters = n_clusters, metric = "manhattan", method = "pam")
    model.fit(one_hot_matrix)
    model.cluster_centers_
    if print_center_features == True:
        print("Prominent features for cluster centers.")
        get_center_features(data, model, state_matrix)
    return model



def auto_cluster(data, include_time = False, only_result = False, 
                 info_table = True, frequency_cutoff = 0.06, plot_search_parameter = True, 
                 plot_clustering = False, info_clustering = False, 
                 plot_params_adjust = False, repeat = 10,
                 redo = None):
    '''Function to do static clustering. Generate clusters of dynophore points automatically.
           input: 
               data: dict - obtained from data pre-processing step
           output: 
               please_manual: list - contain names of superfeatures which suggested to be reclustered with manually specified parameters.
           Attributes:
               include_time: bool - True for adding time component
               only_result: bool - True for only viewing final cluter result
               info_table: bool - True for printing the chosen parameters by add_auto_param() function
               frequency_cutoff: float - parameters for superfeatures of less occuring frequency than frequency_cutoff will not be predicted by add_auto_param() function
               plot_search_parameter: bool - True for plotting the distance distribution of points per superfeature
               plot_clustering: bool - True for plotting the cluster result with automatically picked parameters before parameter adjustment, i.e the first try
               info_clustering: bool - True for printing the information table for clustering step
               plot_params_adjust: bool - True for plotting all clustering figures during params_adjust() programme (it will be a lot)
               repeat: int - repeat time in the first method used in params_adjust() programm
               redo: list - a list of superfeature names which will be re-clustered
           '''
    please_manual = []
    if only_result:
        info_table = False
        plot_search_parameter = False
        plot_clustering = False
        info_clustering = False
        plot_params_adjust = False
    if "params" not in list(data.items())[0][1].keys():
        data = compute.add_auto_param(data, include_time = include_time, 
                                      info_table = info_table, frequency_cutoff = frequency_cutoff, 
                                      plot = plot_search_parameter)
    print("*"*100)
    print("Start clustering")
    data, need_redo = do_cluster(data, include_time = include_time, redo = redo, plot = plot_clustering, info_table = info_clustering)
    print("*"*100)
    print("1. Clustering attemp finished")
    print("*"*100)
    if need_redo:
        data, please_manual = params_adjust(data, need_redo = need_redo, include_time = include_time, 
                                            repeat = repeat, plot = plot_params_adjust)
    # use adjusted parameter to do the clustering -> show result
    print("*"*100)
    print("Result")
    data, need_redo = do_cluster(data, include_time = include_time, redo = None, plot = True)
    
    if please_manual:
        print("Suggest to sepecify parameters for following features. No reliable parameters found.")
        print(please_manual)
    else:
        print("Clustering finished! You may adjust parameters for better result manually.")
        
    return please_manual


def one_line_to_result(dyno_path, pdb_path, dcd_path, output_directory, 
                       include_time = False, data = None, output = True, only_result = False, 
                       info_table = True, frequency_cutoff_param = 0.06, plot_search_parameter = True, 
                       plot_clustering = False, info_clustering = False, 
                       plot_params_adjust = False, repeat = 10, redo = None, 
                       n_clusters = None, print_center_features = False, 
                       noise_cutoff = 0.2, frequency_cutoff = None):
    '''A funtion to choose parameters, cluster, write out trajectories, phamacophores automatically.
       Inputs:
           dyno_path: str - Path to dynophore output folder e.g. "./dynophore_out_2022-06-09_00-15-16-HIV/"
           pdb_path: str - Path to pdb file
           dcd_path: str - Path to MD trajectory
           output_directory: str - Path to output directory
       Attributes:
           include_time: bool - True for adding time component
           data: dict (optional) - obtained from data pre-processing step
           output: bool - True for write out the trajectories, pharmacophores
           only_result:bool - True for only viewing final cluter result
           info_table: bool - True for printing information table in adding parameters and doing clustering
           frequency_cutoff_param: float - parameters for superfeatures of less occuring frequency than frequency_cutoff will not be predicted by add_auto_param() function
           plot_search_parameter: bool - True for plotting the distance distribution of points per superfeature
           plot_clustering: bool - True for plotting the cluster result with automatically picked parameters before parameter adjustment, i.e the first try
           info_clustering: bool - True for printing the information table for clustering step
           plot_params_adjust: bool - True for plotting all clustering figures during params_adjust() programme (it will be a lot)
           repeat: int - repeat time in the first method used in params_adjust() programm
           redo: list - a list of superfeature names which will be re-clustered
           n_clusters: int - number of clusters wanted
           print_center_features: bool - True for printing prominent features for Kemoids cluster centers
           noise_cutoff: float - used in extract_superfeature() for dropping less dominent clusters within each superfeature
           frequency_cutoff: dict - used in extract_superfeature() frequency_cutoff for different superfeatures for dropping noise superfeature in certain cluster
       Outputs:
           data: dict
           model: KMedoids model object -  e.g. KMedoids(method='pam', metric='manhattan', n_clusters=3)
    '''
    if data != None:
        data, need_redo = do_cluster(data, include_time = include_time, redo = redo, plot = True, info_table = True)
    else:
        data = parsers.pre_process(dyno_path, include_time = include_time)
        auto_cluster(data, include_time = include_time, only_result = only_result, 
                                info_table = info_table, frequency_cutoff = frequency_cutoff_param, 
                                plot_search_parameter = plot_search_parameter, 
                                plot_clustering = plot_clustering, info_clustering = info_clustering, 
                                plot_params_adjust = plot_params_adjust, repeat = repeat,
                                redo = redo)
    model = binding_state_cluster(data, n_clusters = n_clusters, print_center_features = print_center_features)
    if output == True:
        write.write_points(data, model, output_directory = output_directory)
        write.write_pharmacophore(data, model, output_directory = output_directory, include_time = include_time, 
                                  frequency_cutoff = frequency_cutoff, noise_cutoff = noise_cutoff, combine = False)
        write.write_pharmacophore(data, model, output_directory = output_directory, include_time = include_time, 
                                  frequency_cutoff = frequency_cutoff, noise_cutoff =noise_cutoff, combine = True)
        write.split_trajectory(model, pdb_path = pdb_path, dcd_path = dcd_path,output_directory = output_directory)
    return data, model