import os
import pickle
import numpy as np

VISION_ROOT = r"C:\Users\MASTER PC\Desktop\vision"

CACHE_DIR = os.path.join(VISION_ROOT, "cache_sift_xy_resize1024_linear")
VOCAB_PATH = os.path.join(VISION_ROOT, "models", "vocab_K2000.pkl")


SPM_DIR = os.path.join(VISION_ROOT, "features_spm_L2_K2000")
os.makedirs(SPM_DIR, exist_ok=True)


def load_descriptors(pkl_path): # load descriptors only from { xy:   (N,2)   desc: (N,128), size: (w,h) }
   
   
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    desc = data["desc"]

    if desc is None:
        return None

    desc = np.asarray(desc, dtype=np.float32)
    if desc.ndim != 2 or desc.shape[1] != 128:
        return None

    return desc


def build_histogram(desc, kmeans, K): #build histogram for a set of descriptors(bcz it will be used in SPM)
  
    histogram = np.zeros(K, dtype=np.float32)

    if desc is None or len(desc) == 0:
        return histogram

    word_ids = kmeans.predict(desc)  # (N,)
    histogram += np.bincount(word_ids, minlength=K).astype(np.float32)

    return histogram




def load_xy_desc_size(pkl_path): #return xy, desc, size from SIFT cache (xy are needed for matching points into grids)
   
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    xy = data.get("xy", None)
    desc = data.get("desc", None)
    size = data.get("size", None)

    if desc is None or xy is None or size is None:
        return None, None, None

    xy = np.asarray(xy, dtype=np.float32)
    desc = np.asarray(desc, dtype=np.float32)

    # Basic shape checks
    if xy.ndim != 2 or xy.shape[1] != 2:
        return None, None, None
    if desc.ndim != 2 or desc.shape[1] != 128:
        return None, None, None

    return xy, desc, size


def l2_normalize(vec, eps=1e-12):
    n = np.sqrt(np.sum(vec * vec))
    if n < eps:
        return vec
    return vec / n


def build_spm_feature(xy, desc, size, kmeans, K, levels=(0, 1, 2)):
    """
    Spatial Pyramid Matching feature (RAW counts, no normalization).

    levels:
      0 -> 1x1
      1 -> 2x2
      2 -> 4x4

    Output dimension: (1 + 4 + 16) * K = 21*K
    """
    # handle empty
    if desc is None or xy is None or len(desc) == 0:
        return np.zeros((sum((2**l)**2 for l in levels) * K,), dtype=np.float32)

    w, h = size
    if w is None or h is None or w <= 0 or h <= 0:
        return np.zeros((sum((2**l)**2 for l in levels) * K,), dtype=np.float32)

    parts = []

    for l in levels:
        bins = 2 ** l
        x = np.clip(xy[:, 0], 0, w - 1e-6)
        y = np.clip(xy[:, 1], 0, h - 1e-6)

        cx = (x * bins / w).astype(np.int32)
        cy = (y * bins / h).astype(np.int32)
        cell_id = cy * bins + cx

        for c in range(bins * bins):
            idx = np.where(cell_id == c)[0]
            if len(idx) == 0:
                parts.append(np.zeros(K, dtype=np.float32))
            else:
                parts.append(build_histogram(desc[idx], kmeans, K))

    return np.concatenate(parts, axis=0).astype(np.float32)



def main(limit=None):
    # load vocabulary (k-means model)
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    kmeans = vocab["kmeans"]
    K = vocab["K"]

    pkl_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pkl")]
    pkl_files.sort()

    if limit is not None:
        pkl_files = pkl_files[:limit]

    print("Building SPM features for:", len(pkl_files), "images")
    print("K (vocab size) =", K)
    print("SPM levels = (0,1,2) => dim =", 21 * K)

    for i, fname in enumerate(pkl_files, start=1):
        pkl_path = os.path.join(CACHE_DIR, fname)

        xy, desc, size = load_xy_desc_size(pkl_path)
        spm_vec = build_spm_feature(xy, desc, size, kmeans, K, levels=(0, 1, 2))


        out_name = fname.replace(".pkl", ".npy")
        out_path = os.path.join(SPM_DIR, out_name)
        np.save(out_path, spm_vec)

        if i == 1:
            print("Example SPM shape:", spm_vec.shape)
            print("First 10 values:", spm_vec[:10])

        if i % 50 == 0:
            print("Processed", i, "/", len(pkl_files))

    print("Done. SPM features saved in:", SPM_DIR)


if __name__ == "__main__":
    main(limit=None)
