import os
import numpy as np

VISION_ROOT = r"C:\Users\MASTER PC\Desktop\vision"
TFIDF_DIR = os.path.join(VISION_ROOT, "features_spm_L2_K2000_tfidf")


DB_PATH = os.path.join(VISION_ROOT, "models", "db_spm_tfidf_L2.npz")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def build_db(tfidf_dir): #build database from TFIDF vectors for each image 
    files = sorted([
        f for f in os.listdir(tfidf_dir)
        if f.endswith(".npy") and f.lower() != "idf.npy"
    ])
    if not files:
        raise RuntimeError("No image vectors found in TFIDF_DIR")

    # load first to get dimension to generalize the code
    v0 = np.load(os.path.join(tfidf_dir, files[0])).astype(np.float32)
    if v0.ndim != 1:
        raise RuntimeError(f"Expected 1D vector in {files[0]}, got shape {v0.shape}")

    D = int(v0.shape[0])
    N = len(files)

    X = np.zeros((N, D), dtype=np.float32)

    for i, fname in enumerate(files):
        v = np.load(os.path.join(tfidf_dir, fname)).astype(np.float32)

        if v.ndim != 1:
            raise RuntimeError(f"{fname} is not 1D. Got shape {v.shape}")
        if v.shape[0] != D:
            raise RuntimeError(f"Dim mismatch {fname}: got {v.shape[0]}, expected {D}")

        X[i] = v

        if (i + 1) % 200 == 0:
            print("Loaded", i + 1, "/", N)

    

    return X, files, D, N


if __name__ == "__main__":
    X, files, D, N = build_db(TFIDF_DIR)

    # Save
    np.savez_compressed(DB_PATH, X=X, files=np.array(files), D=np.array([D]), N=np.array([N]))

    print("Saved DB to:", DB_PATH)
    print("X shape:", X.shape)





