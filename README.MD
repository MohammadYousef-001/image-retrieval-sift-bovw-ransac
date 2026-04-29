Dataset:
Oxford Buildings Dataset (Oxford5k)

Image data:
Downloaded from Kaggle:
https://www.kaggle.com/datasets/skylord/oxbuildings

Ground truth and evaluation files:
Provided by the original Oxford Visual Geometry Group:
https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/

Expected folder structure:
- images/
- ground_truth/
- cache_sift_xy_resize1024_linear/
- features_spm_L2_K2000/
- features_spm_L2_K2000_tfidf/
- models/

How to run:
- Baseline retrieval: python main.py
- RANSAC-enhanced retrieval: python mainR.py
