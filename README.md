# Reveal Hidden Interaction Pattern and Seperate Pharmacophore for each Bind State

A software developed to recognize target-complex binding modes based on interaction pattern recorded in dynamic pharmacophores. Condense pharmacophone and associated dynophore points will be written out, as well as the MD trajectory for each binding mode.

- **Demo:** [ZIKV_Protease](https://nbviewer.org/github/ChristyLau/thesis/blob/main/Demo-ZIKV-time.ipynb)



## Requirement
- cnnclustering
- sklearn
- matplotlib
- mdanalysis
- numpy
- pandas
- tqdm
- os


## Output


## Workflow
This is a two-stage clustering:

  - a) Cluster individual superfeatures to obtained states within each interaction (CommonNNClustering)
  - b) Cluster all features together in a categorical state space (K-Medoids)
