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

## 3. YOLO Model Validation

After training a YOLO model, we proceed to perform **note detection** on music sheets.

### 3.1 Direct Validation

Initially, we applied the trained model directly to the entire music sheet without any pre-processing. However, the results were unsatisfactory: the **mAP@50 on the full DeepScores validation set was only around 67**.

![Direct Validation Result](./attachments/direct_validation.png)

This underperformance is expected. Dense object detection with many small and tightly packed symbols (like musical notes) is inherently challenging for YOLO and most general-purpose object detectors. 

### 3.2 Pairwise Staffline Detection Strategy

To address this, we implemented a more targeted detection strategy:
- We first **detect staff lines** and **split each sheet into staffline pairs (10-line segments)**.
- Each pair is then **cropped, resized, and individually processed** by YOLO.
- The results are stitched back to the full image space.

This approach leverages the local structure of music sheets and greatly simplifies the detection problem for the model. Here's the resulting performance:

![Pairwise Validation Result](./attachments/pairwise_validation.png)

We achieved a significant performance boost:  
**mAP@50 increased from 67 → 86.5**  
(using the exact same trained model as before)

### 3.3 Visual Comparison

Below is a side-by-side visualization of detection results using:
- **Left:** Direct inference on full sheet
- **Right:** Pairwise cropped inference

![Comparison](./attachments/comparison.png)

As seen, the **pairwise detection almost perfectly identifies notes** between the staff lines, far surpassing the direct detection baseline.

---

### 3.4 How to Validate

- To validate model performance on deepscore validation set:  
  **Run `yolo_validation.ipynb`**
  
- To visualize detection outputs across multiple images:  
  **Run `yolo_visualization.ipynb`**

## 4. Staffline Detection

In the previous section on pairwise detection, we mentioned that the image was segmented by **staffline pairs**. The foundation of this segmentation lies in **staffline detection** — accurately identifying horizontal lines on music sheets.

Our detection pipeline includes:
- Converting the image to grayscale
- Applying a 1D horizontal convolution filter
- Filtering based on pixel intensity and black-pixel ratio
- Grouping detected lines into sets of 5 (representing a staff)

This staffline information allows us to **segment the music sheet into cropped regions of interest**, improving both efficiency and detection accuracy.

Below is a sample of staffline_detection (without yolo prediction)
![Staffline Detection](./attachments/staffline_detection.png)

To directly see the detected stafflines overlaid on the original image, you can run:

```bash
python staffline_detection.py
```
