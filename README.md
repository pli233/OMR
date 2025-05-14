# Optical Music Recognition (OMR) with YOLO and DeepScoresV2 Dense

This project aims to detect musical symbols in sheet music images using the YOLO object detection framework, as a first step toward converting printed music into playable MIDI files.

---

## 1. Dataset Preparation

We use the **DeepScoresV2 Dense** dataset for training and evaluation.

- Official source: [Zenodo Record](https://zenodo.org/records/4012193)
- Backup copy: [Google Drive Folder](https://drive.google.com/drive/folders/1Fh5MDLxmB_od7o7MvaRp8b55Dn7dbvi0)

> 📦 **Note:** Please download and extract `ds2_dense.tar.gz` before proceeding.

### Preprocessing Instructions

Before training, you must clean and reformat the dataset to be compatible with YOLO:

- Annotations must be converted into YOLO format.
- Images and labels must be organized into `images/train`, `images/test`, `labels/train`, and `labels/test`.

Run the notebook `prepare_ds2_for_yolo.ipynb` to complete this step.

---

## 2. YOLO Model Training

The first stage in our OMR pipeline is training a YOLO model to detect music symbols.

To start training run:

```bash
python yolo_train.py
```
Your dataset folder (ds2_dense/) must contain:
- images/train/ and images/test/
- labels/train/ and labels/test/

Your OMR directory must have: 
- deepscores.yaml

If any of above files are missing, re-run `prepare_ds2_for_yolo.ipynb` before training.

## 3. 