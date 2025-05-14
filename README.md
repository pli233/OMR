# Optical Music Recognition



# 1. Dataset Prepration

In this project, we use deepscore v2 dense dataset, it can be find at https://zenodo.org/records/4012193.
We also attach a copy of ds2_dense.tar.gz in our google cloud

To make sure that yolo model can be trained properly, we have done some data clean up and structure adjustment.
Please run all commands in prepare_ds2_for_yolo.ipynb before start running other scripts of our project

# 2. YOLO model training

Our first step to create midi is train a YOLO model
run yolo_train.py to train the model
please make sure that your ds2_dense have ds2_dense/labels directory. It includes the annotation format meets the requiremnt of YOLO
If u dont have it, run prepare_ds2_for_yolo.ipynb first

# 3.