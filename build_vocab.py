import os
import random
import pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans

VISION_ROOT = r"C:\Users\MASTER PC\Desktop\vision"

# IMPORTANT: point to the NEW cache that contains {"xy","desc","size"}
CACHE_DIR = os.path.join(VISION_ROOT, "cache_sift_xy_resize1024_linear")

MODEL_DIR = os.path.join(VISION_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

VOCAB_PATH = os.path.join(MODEL_DIR, "vocab_K2000.pkl")

K = 2000
MAX_DESC_TOTAL = 300_000 # number of descriptors to use for k-means
MAX_DESC_PER_IMAGE = 400 
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

def load_descriptors(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    
    desc = data["desc"] #load only descriptors from SIFT cache format

    if desc is None:
        return None

    desc = np.asarray(desc, dtype=np.float32)
    if desc.ndim != 2 or desc.shape[1] != 128:
        return None

    return desc


def main():
    pkl_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pkl")]
    pkl_files.sort()
    random.shuffle(pkl_files)

    sampled = []
    total = 0

    for fname in pkl_files:
        path = os.path.join(CACHE_DIR, fname)
        desc = load_descriptors(path)

        if desc is None or len(desc) == 0:
            continue

        # sample per image to avoid domination by one textured image(bcz it has large number of descriptors)
        if len(desc) > MAX_DESC_PER_IMAGE:
            idx = np.random.choice(len(desc), MAX_DESC_PER_IMAGE, replace=False)
            desc = desc[idx]

        sampled.append(desc)
        total += len(desc)

        if total >= MAX_DESC_TOTAL:
            break

    if len(sampled) == 0:
        raise RuntimeError("No descriptors loaded. Check your cache folder.")

    X = np.vstack(sampled)[:MAX_DESC_TOTAL]
    print("Descriptors used for k-means:", X.shape)

    kmeans = MiniBatchKMeans(
        n_clusters=K,
        batch_size=10_000,
        random_state=SEED,
        verbose=0
    )
    kmeans.fit(X)

    with open(VOCAB_PATH, "wb") as f:
        pickle.dump({"K": K, "kmeans": kmeans}, f)

    print("Saved vocabulary to:", VOCAB_PATH)

if __name__ == "__main__":
    main()
