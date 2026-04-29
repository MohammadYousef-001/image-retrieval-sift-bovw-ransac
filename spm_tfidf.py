import os
import numpy as np

VISION_ROOT = r"C:\Users\MASTER PC\Desktop\vision"
SPM_RAW_DIR = os.path.join(VISION_ROOT, "features_spm_L2_K2000")          # your current output
SPM_TFIDF_DIR = os.path.join(VISION_ROOT, "features_spm_L2_K2000_tfidf")  # new output
os.makedirs(SPM_TFIDF_DIR, exist_ok=True)

K = 2000
NUM_REGIONS = 21  # levels (0,1,2) 1 + 4 + 16

def l2_normalize(vec, eps=1e-12): #we normalize after tfidf weighting
    n = np.sqrt(np.sum(vec * vec))
    if n < eps:
        return vec
    return vec / n

def compute_idf(spm_raw_dir, K, num_regions): #global idf computation(depemds on how many images the word appears in at least once)
    files = [f for f in os.listdir(spm_raw_dir) if f.endswith(".npy")]
    files.sort()
    N = len(files)

    df = np.zeros(K, dtype=np.int64) #number of images the word j appears in at least once

    for fname in files:
        v = np.load(os.path.join(spm_raw_dir, fname)).astype(np.float32)

        # reshape into (21, K)
        v = v.reshape(num_regions, K)

        # word is "present in image" if it appears in ANY region
        present = (np.sum(v, axis=0) > 0)
        df += present.astype(np.int64)

    # edge case to avoid division by zero (very rare words)
    idf = np.log((N + 1.0) / (df + 1.0)).astype(np.float32) + 1.0
    return idf, files

def apply_tfidf(spm_raw_dir, spm_out_dir, idf, files, K, num_regions, use_tf=True):
    idf_full = np.tile(idf, num_regions).astype(np.float32)  # repeat for each region

    for i, fname in enumerate(files, start=1):
        v = np.load(os.path.join(spm_raw_dir, fname)).astype(np.float32)

        if use_tf: # term frequency in image
            # TF: normalize by total counts in the whole pyramid (like n_id / n_d)
            s = float(np.sum(v))
            if s > 0:
                v = v / s

        # tfidf weighting
        v = v * idf_full

        # 2nd normnormalize for cosine similarity retrieval
        v = l2_normalize(v)

        np.save(os.path.join(spm_out_dir, fname), v)

        if i % 200 == 0:
            print("TF-IDF processed", i, "/", len(files))

def main():
    idf, files = compute_idf(SPM_RAW_DIR, K, NUM_REGIONS)
    np.save(os.path.join(SPM_TFIDF_DIR, "idf.npy"), idf)
    apply_tfidf(SPM_RAW_DIR, SPM_TFIDF_DIR, idf, files, K, NUM_REGIONS, use_tf=True)
    print("Done. TF-IDF SPM features saved in:", SPM_TFIDF_DIR)

if __name__ == "__main__":
    main()
