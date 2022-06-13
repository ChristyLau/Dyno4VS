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

## Workflow
This is a two-stage clustering:

  - a) Cluster individual superfeatures to obtained states within each interaction (CommonNNClustering)
  - b) Cluster all features together in a categorical state space (K-Medoids)


## Output
For each binding mode
- Dynophore points in PML file (noise included)
- Dynophore in PML file (noise dropped)
- Dynophore and associated points in PML file (noise dropped)
- MD trajectory

## Analysis
- Bar code plot: frames belong to each cluster
- Stacked bar plot: frequency of each state within each superfeature for all binding modes
- Radar plot: frequency of superfeature for all binding modes
