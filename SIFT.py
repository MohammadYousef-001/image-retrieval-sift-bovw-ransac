import os
import cv2
import pickle
import numpy as np


IMAGES_DIR = r"C:\Users\MASTER PC\Desktop\vision\images"


CACHE_DIR = r"C:\Users\MASTER PC\Desktop\vision\cache_sift_xy_resize1024_linear"

os.makedirs(CACHE_DIR, exist_ok=True)


sift = cv2.SIFT_create()


def resize_max_side(img, max_side=1024): #resize image longest size if it is larger than max_side to max_side
   
    h, w = img.shape[:2]
    m = max(h, w)

    if m <= max_side:
        return img

    scale = max_side / float(m)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def extract_sift_xy_desc(image_path): #extract sift descriptors and x,y and size 
    """
    Extract SIFT keypoint (x,y) and descriptors from one image (after resizing).
    Returns:
        data dict or None
        data = {
            "xy": (N,2) float32,
            "desc": (N,128) float32,
            "size": (w,h) int
        }
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = resize_max_side(img, max_side=1024)
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) == 0:
        return {"xy": None, "desc": None, "size": (w, h)}

    # (x, y) for each keypoint
    xy = np.array([kp.pt for kp in keypoints], dtype=np.float32)  # shape (N,2)
    desc = descriptors.astype(np.float32)  # shape (N,128)

    return {"xy": xy, "desc": desc, "size": (w, h)}


def save_sift(path, data):#save sift data to diskk
    with open(path, "wb") as f:
        pickle.dump(data, f)


def main(limit=None):
    images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")]
    images.sort()

    if limit is not None:
        images = images[:limit]

    total = len(images)
    print("Images to process:", total)

    for i, fname in enumerate(images, start=1):
        img_path = os.path.join(IMAGES_DIR, fname)
        data = extract_sift_xy_desc(img_path)

        out_path = os.path.join(CACHE_DIR, fname.replace(".jpg", ".pkl"))
        save_sift(out_path, data)

        if i % 50 == 0 or i == total:
            print(f"Processed {i}/{total}")

    print("Done. SIFT (xy+desc) cached in:", CACHE_DIR)


if __name__ == "__main__":
    main(limit=None)
