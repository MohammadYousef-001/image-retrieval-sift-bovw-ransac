## Dataset

This project uses the **Oxford Buildings Dataset (Oxford5k)**.

### Image Data
The image dataset was downloaded from Kaggle:  
https://www.kaggle.com/datasets/skylord/oxbuildings

### Ground Truth and Evaluation Files
The ground truth and evaluation files were obtained from the original Oxford Visual Geometry Group (VGG):  
https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/

## Expected Folder Structure

```text
project-folder/
│
├── images/
├── ground_truth/
├── cache_sift_xy_resize1024_linear/
├── features_spm_L2_K2000/
├── features_spm_L2_K2000_tfidf/
├── models/
│
├── main.py
├── mainR.py
└── README.md
