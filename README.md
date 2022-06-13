# Reveal Hidden Interaction Pattern and Seperate Pharmacophore for each Bind State

A software developed to recognize target-complex binding modes based on interaction pattern recorded in dynamic pharmacophores. Condense pharmacophone and associated dynophore points will be written out, as well as the MD trajectory for each binding mode.

- **Demo:** [ZIKV_Protease](https://nbviewer.org/github/ChristyLau/thesis/blob/main/Demo-ZIKV-time.ipynb)



## Requirement
- cnnclustering
- sklearn
- sklearn_extra
- matplotlib
- mdanalysis
- numpy
- pandas
- tqdm
- os


## Install

```$ git clone https://github.com/ChristyLau/thesis.git```

```$ conda env update --file requirement.yaml```


## Workflow
This is a two-stage clustering:

  - a) Cluster individual superfeatures to obtained states within each interaction (CommonNNClustering)
  - b) Cluster all features together in a categorical state space (K-Medoids)


## Output
For each binding mode (mode 2 as an example)
- Dynophore points in PML file (noise included)

![image](https://github.com/ChristyLau/thesis/blob/main/fig/points_with_noise.png) 
- Dynophore in PML file (noise dropped)

![image](https://github.com/ChristyLau/thesis/blob/main/fig/dyno_without_noise.png) 
- Dynophore and associated points in PML file (noise dropped)

![image](https://github.com/ChristyLau/thesis/blob/main/fig/dyno%2Bpoints_without_noise.png)  
- MD trajectory

## Analysis
- Bar code plot: frames belong to each cluster

![image](https://github.com/ChristyLau/thesis/blob/main/fig/bar_code.png)  
- Stacked bar plot: frequency of each state within each superfeature for all binding modes

![image](https://github.com/ChristyLau/thesis/blob/main/fig/stacked_bar.png)  
- Radar plot: frequency of superfeature for all binding modes

![image](https://github.com/ChristyLau/thesis/blob/main/fig/radar.png)  
