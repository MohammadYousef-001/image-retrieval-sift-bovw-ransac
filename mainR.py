import os
import time
import numpy as np
import pickle
import cv2

VISION_ROOT = r"C:\Users\MASTER PC\Desktop\vision"
IMAGES_DIR = r"C:\Users\MASTER PC\Desktop\vision\images"
GT_DIR = r"C:\Users\MASTER PC\Desktop\vision\ground_truth"
DB_PATH = os.path.join(VISION_ROOT, "models", "db_spm_tfidf_L2.npz")


CACHE_DIR = os.path.join(VISION_ROOT, "cache_sift_xy_resize1024_linear")



class Query:
    def __init__(self, query_id, query_image, good, ok, junk):
        self.query_id = query_id
        self.query_image = query_image  # with .jpg
        self.good = good
        self.ok = ok
        self.junk = junk

    def __repr__(self):
        return f"Query({self.query_id})"


def load_images(images_dir):
    return [f.replace(".jpg", "") for f in os.listdir(images_dir) if f.endswith(".jpg")]


def load_id_list(file_path):
    ids = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def load_queries(gt_dir):
    queries = []
    for file in os.listdir(gt_dir):
        if file.endswith("_query.txt"):
            query_id = file.replace("_query.txt", "")
            query_path = os.path.join(gt_dir, file)

            with open(query_path, "r") as f:
                line = f.readline().strip()
                query_image = line.split()[0] + ".jpg"

            good = load_id_list(os.path.join(gt_dir, query_id + "_good.txt"))
            ok   = load_id_list(os.path.join(gt_dir, query_id + "_ok.txt"))
            junk = load_id_list(os.path.join(gt_dir, query_id + "_junk.txt"))

            queries.append(Query(query_id, query_image, good, ok, junk))
    return queries


def load_db(db_path):
    db = np.load(db_path, allow_pickle=True)
    X = db["X"].astype(np.float32)
    files = db["files"].astype(str).tolist()  # ["xxx.npy", ...]
    return X, files


def build_id_to_index(files):
    return {os.path.splitext(f)[0]: i for i, f in enumerate(files)}


def normalize_id(img_id, id_to_idx):
    if img_id in id_to_idx:
        return img_id

    if img_id.startswith("oxc1_"):
        img_id2 = img_id[len("oxc1_"):]
        if img_id2 in id_to_idx:
            return img_id2

    if "_" in img_id:
        img_id2 = img_id.split("_", 1)[1]
        if img_id2 in id_to_idx:
            return img_id2

    return None



def rank_by_cosine(X, files, qvec):
    scores = X @ qvec
    order = np.argsort(-scores)
    return [os.path.splitext(files[i])[0] for i in order]


def remove_junk(ranked_ids, junk):
    junk = set(junk)
    return [rid for rid in ranked_ids if rid not in junk]


def average_precision(ranked_ids, pos_set):
    hits = 0
    s = 0.0
    for i, r in enumerate(ranked_ids, start=1):
        if r in pos_set:
            hits += 1
            s += hits / i
    return s / len(pos_set) if pos_set else 0.0


def precision_at_k(ranked_ids, pos_set, k):
    topk = ranked_ids[:k]
    hits = sum(1 for r in topk if r in pos_set)
    return hits / k


def recall_at_k(ranked_ids, pos_set, k):
    topk = ranked_ids[:k]
    hits = sum(1 for r in topk if r in pos_set)
    return hits / len(pos_set) if pos_set else 0.0



def load_sift_from_cache(image_id):
    """
    Loads (xy, desc) from cache .pkl
    expected keys: xy (N,2), desc (N,128)
    """
    pkl_path = os.path.join(CACHE_DIR, image_id + ".pkl")
    if not os.path.exists(pkl_path):
        return None, None

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    xy = data.get("xy", None)
    desc = data.get("desc", None)

    if xy is None or desc is None:
        return None, None

    xy = np.asarray(xy, dtype=np.float32)
    desc = np.asarray(desc, dtype=np.float32)

    if xy.ndim != 2 or xy.shape[1] != 2:
        return None, None
    if desc.ndim != 2 or desc.shape[1] != 128:
        return None, None

    return xy, desc


def ransac_inliers(qid, cid, ratio=0.75, min_good=12, ransac_thresh=5.0): #returns number of RANSAC inliers between query image and candidate
   
    q_xy, q_desc = load_sift_from_cache(qid)
    c_xy, c_desc = load_sift_from_cache(cid)

    if q_desc is None or c_desc is None:
        return 0
    if len(q_desc) < 2 or len(c_desc) < 2:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(q_desc, c_desc, k=2)

    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < min_good:
        return 0

    src = np.float32([q_xy[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([c_xy[m.trainIdx] for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if mask is None:
        return 0

    return int(mask.ravel().sum())


def rerank_topN_with_ransac(qid, ranked_ids, topN=100):
   
    cand = ranked_ids[:topN]
    scored = [(cid, ransac_inliers(qid, cid)) for cid in cand]
    scored.sort(key=lambda x: x[1], reverse=True)
    reranked = [cid for cid, _ in scored]
    return reranked + ranked_ids[topN:]



if __name__ == "__main__":


    t_load0 = time.perf_counter()

    images = load_images(IMAGES_DIR)
    queries = load_queries(GT_DIR)

    X, files = load_db(DB_PATH)
    id_to_idx = build_id_to_index(files)

    t_load1 = time.perf_counter()
    load_time = t_load1 - t_load0

    print("\nImages:", len(images))
    print("Queries:", len(queries))
    print("DB loaded. X shape:", X.shape)
    print(f"LOAD TIME (images+queries+DB): {load_time:.3f} sec")

    
    K = int(input("\nEnter K (e.g. 5, 10, 20): "))

    print("\nChoose mode:")
    print("1 = Return Top-K for ALL queries")
    print("2 = Return Top-K for ONE query")
    mode = int(input("Your choice: "))

    qnum = None
    if mode == 2:
        qnum = int(input(f"\nEnter query number (1–{len(queries)}): "))

    
    TOPN = int(input("\nRANSAC Top-N candidates to re-rank (e.g. 50, 100, 200): "))


    t_run0 = time.perf_counter()

    def process_query(q, q_index):
    
        raw_qid = os.path.splitext(q.query_image)[0]
        qid = normalize_id(raw_qid, id_to_idx)
        if qid is None:
            raise RuntimeError(f"Query image '{raw_qid}' not found in DB")

      
        good = [normalize_id(x, id_to_idx) for x in q.good]
        ok   = [normalize_id(x, id_to_idx) for x in q.ok]
        junk = [normalize_id(x, id_to_idx) for x in q.junk]

        good = [x for x in good if x is not None]
        ok   = [x for x in ok if x is not None]
        junk = [x for x in junk if x is not None]

        pos_set = set(good) | set(ok)

      
        t_ret0 = time.perf_counter()
        qvec = X[id_to_idx[qid]]
        ranked_ids = rank_by_cosine(X, files, qvec)
        ranked_ids = remove_junk(ranked_ids, junk)
        t_ret1 = time.perf_counter()
        retrieval_time = t_ret1 - t_ret0

       
        t_ran0 = time.perf_counter()
        ranked_ransac = rerank_topN_with_ransac(qid, ranked_ids, topN=TOPN)
        t_ran1 = time.perf_counter()
        ransac_time = t_ran1 - t_ran0

      
        t_eval0 = time.perf_counter()
        P = precision_at_k(ranked_ransac, pos_set, K)
        R = recall_at_k(ranked_ransac, pos_set, K)
        AP = average_precision(ranked_ransac, pos_set)
        t_eval1 = time.perf_counter()
        eval_time = t_eval1 - t_eval0

        print("\n===================================")
        print(f"QUERY #{q_index+1}: {q.query_id}")
        print(f"TOP-K = {K}   |   RANSAC Top-N = {TOPN}")
        print("-----------------------------------")
        print(f"PRECISION@{K} = {P:.4f}")
        print(f"RECALL@{K}    = {R:.4f}")
        print(f"AP            = {AP:.4f}")
        print("-----------------------------------")
        print(f"RETRIEVAL TIME: {retrieval_time*1000:.2f} ms")
        print(f"RANSAC TIME:    {ransac_time*1000:.2f} ms")
        print(f"EVAL TIME:      {eval_time*1000:.2f} ms")
        print("===================================")

        return P, R, AP, (retrieval_time + ransac_time), eval_time

  
    if mode == 1:
        P_list, R_list, AP_list = [], [], []
        total_times, eval_times = [], []

        for i, q in enumerate(queries):
            P, R, AP, t_total, t_eval = process_query(q, i)
            P_list.append(P)
            R_list.append(R)
            AP_list.append(AP)
            total_times.append(t_total)
            eval_times.append(t_eval)

        print("\n========== AVERAGE OVER ALL QUERIES (RANSAC) ==========")
        print(f"MEAN PRECISION@{K} = {np.mean(P_list):.4f}")
        print(f"MEAN RECALL@{K}    = {np.mean(R_list):.4f}")
        print(f"mAP                = {np.mean(AP_list):.4f}")
        print("---------------------------------------------")
        print(f"AVG (retrieval+ransac) TIME/query: {np.mean(total_times)*1000:.2f} ms")
        print(f"AVG EVAL TIME/query:               {np.mean(eval_times)*1000:.2f} ms")

   
    elif mode == 2:
        q = queries[qnum - 1]
        process_query(q, qnum - 1)

    else:
        print("Invalid mode.")

   
    t_run1 = time.perf_counter()
    run_time = t_run1 - t_run0

    print("\n========== TIME SUMMARY ==========")
    print(f"RUN TIME (excluding user input): {run_time:.3f} sec")
    print(f"LOAD TIME:                       {load_time:.3f} sec")
    print(f"TOTAL SYSTEM TIME:               {(load_time + run_time):.3f} sec")
