import os
import time
import numpy as np

VISION_ROOT = r"C:\Users\MASTER PC\Desktop\vision"
IMAGES_DIR = r"C:\Users\MASTER PC\Desktop\vision\images"
GT_DIR = r"C:\Users\MASTER PC\Desktop\vision\ground_truth"
DB_PATH = os.path.join(VISION_ROOT, "models", "db_spm_tfidf_L2.npz")


#query class ( save query id, query image name, good, ok, junk lists)
class Query:
    def __init__(self, query_id, query_image, good, ok, junk):
        self.query_id = query_id
        self.query_image = query_image  # with .jpg
        self.good = good
        self.ok = ok
        self.junk = junk

    def __repr__(self):
        return f"Query({self.query_id})"



def load_images(images_dir): #load imgs from images dir
    return [f.replace(".jpg", "") for f in os.listdir(images_dir) if f.endswith(".jpg")]


def load_id_list(file_path): #load ids from a text file (one id per line) (used for good, ok, junk lists)
    ids = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def load_queries(gt_dir): #load queries 
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


def load_db(db_path): #load db matrix and file names (file names are needed to map indices to image ids)
    db = np.load(db_path, allow_pickle=True)
    X = db["X"].astype(np.float32)
    files = db["files"].astype(str).tolist()  # ["xxx.npy", ...]
    return X, files


def build_id_to_index(files): #to get index of an image id in the db matrix to access its vector
    return {os.path.splitext(f)[0]: i for i, f in enumerate(files)}



def normalize_id(img_id, id_to_idx):
    if img_id in id_to_idx:
        return img_id
    if img_id.startswith("oxc1_"):
        img_id = img_id[len("oxc1_"):]
        if img_id in id_to_idx:
            return img_id
    if "_" in img_id:
        img_id = img_id.split("_", 1)[1]
        if img_id in id_to_idx:
            return img_id
    return None



def rank_by_cosine(X, files, qvec): 
    scores = X @ qvec
    order = np.argsort(-scores)
    return [os.path.splitext(files[i])[0] for i in order]


def remove_junk(ranked_ids, junk):#must be done oxford style evaluation protocol
    junk = set(junk)
    return [rid for rid in ranked_ids if rid not in junk]


def precision_recall_at_k(ranked_ids, pos_set, k):
    topk = ranked_ids[:k]
    hits = sum(1 for r in topk if r in pos_set)
    precision = hits / k
    recall = hits / len(pos_set) if pos_set else 0.0
    return precision, recall


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



if __name__ == "__main__":
    import time

   
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

       
        t_eval0 = time.perf_counter()
        P = precision_at_k(ranked_ids, pos_set, K)
        R = recall_at_k(ranked_ids, pos_set, K)
        AP = average_precision(ranked_ids, pos_set)
        t_eval1 = time.perf_counter()
        eval_time = t_eval1 - t_eval0

        print("\n===================================")
        print(f"QUERY #{q_index+1}: {q.query_id}")
        print(f"TOP-K = {K}")
        print("-----------------------------------")
        print(f"PRECISION@{K} = {P:.4f}")
        print(f"RECALL@{K}    = {R:.4f}")
        print(f"AP            = {AP:.4f}")
        print("-----------------------------------")
        print(f"RETRIEVAL TIME: {retrieval_time*1000:.2f} ms")
        print(f"EVAL TIME:      {eval_time*1000:.2f} ms")
        print("===================================")

        return P, R, AP, retrieval_time, eval_time

  
    if mode == 1:
        P_list, R_list, AP_list = [], [], []
        ret_times, eval_times = [], []

        for i, q in enumerate(queries):
            P, R, AP, rt, et = process_query(q, i)
            P_list.append(P)
            R_list.append(R)
            AP_list.append(AP)
            ret_times.append(rt)
            eval_times.append(et)

        print("\n========== AVERAGE OVER ALL QUERIES ==========")
        print(f"MEAN PRECISION@{K} = {np.mean(P_list):.4f}")
        print(f"MEAN RECALL@{K}    = {np.mean(R_list):.4f}")
        print(f"mAP                = {np.mean(AP_list):.4f}")
        print("---------------------------------------------")
        print(f"AVG RETRIEVAL TIME/query: {np.mean(ret_times)*1000:.2f} ms")
        print(f"AVG EVAL TIME/query:      {np.mean(eval_times)*1000:.2f} ms")

    
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
