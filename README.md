# PCPR-Net
该代码改自PointNetVlad[here](https://github.com/mikacuy/pointnetvlad)， 并参考借鉴了LPD-Net[here](https://github.com/Suoivy/LPD-net)

### Pre-Requisites
* PyTorch 1.4.0
* tensorboardX

## Benchmark Datasets
The benchmark datasets introdruced in this work can be downloaded [here](https://drive.google.com/open?id=1H9Ep76l8KkUpwILY-13owsEMbVCYTmyx), which created by PointNetVLAD for point cloud based retrieval for place recognition from the open-source [Oxford RobotCar](https://robotcar-dataset.robots.ox.ac.uk/). Details can be found in [PointNetVLAD](https://arxiv.org/abs/1804.03492).
* All submaps are in binary file format
* Ground truth GPS coordinate of the submaps are found in the corresponding csv files for each run
* Filename of the submaps are their timestamps which is consistent with the timestamps in the csv files
* Use CSV files to define positive and negative point clouds
* All submaps are preprocessed with the road removed and downsampled to 4096 points

### Oxford Dataset
* 45 sets in total of full and partial runs
* Used both full and partial runs for training but only used full runs for testing/inference
* Training submaps are found in the folder "pointcloud_20m_10overlap/" and its corresponding csv file is "pointcloud_locations_20m_10overlap.csv"
* Training submaps are not mutually disjoint per run
* Each training submap ~20m of car trajectory and subsequent submaps are ~10m apart
* Test/Inference submaps found in the folder "pointcloud_20m/" and its corresponding csv file is "pointcloud_locations_20m.csv"
* Test/Inference submaps are mutually disjoint

### Dataset set-up
Download the zip file of the benchmark datasets found [here](https://drive.google.com/open?id=1H9Ep76l8KkUpwILY-13owsEMbVCYTmyx).


### Generate pickle files
```
cd generating_queries/

# For training tuples in our baseline network
python generate_training_tuples_baseline.py

# For training tuples in our refined network
# python generate_training_tuples_refine.py

# For network evaluation
python generate_test_sets.py
```

### Train
```
python train_pointnetvlad.py
```

### Evaluate
```
python evaluate.py
```

Take a look atinitPara for more parameters
