import os
import cv2
import shutil
import hashlib
import numpy as np
import streamlit as st

from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from sklearn.model_selection import train_test_split

RAW_IMAGE_DIR = "dataset/train/images"
RAW_LABEL_DIR = "dataset/train/labels"

PROCESSED_DIR = "processed_dataset"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

os.makedirs(PROCESSED_DIR, exist_ok=True)

clean_images = []

for root, dirs, files in os.walk(RAW_IMAGE_DIR):

    for file in files:

        if not any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
            continue

        file_path = os.path.join(root, file)

        try:

            img = cv2.imread(file_path)

            if img is None:
                continue

            h, w = img.shape[:2]

            if h < 100 or w < 100:
                continue

            clean_images.append(file)

        except:
            continue

print(f"Clean Images: {len(clean_images)}")

hashes = {}
unique_images = []

for file in clean_images:

    file_path = os.path.join(RAW_IMAGE_DIR, file)

    with open(file_path, "rb") as f:

        file_hash = hashlib.md5(f.read()).hexdigest()

    if file_hash not in hashes:

        hashes[file_hash] = file
        unique_images.append(file)

print(f"Unique Images: {len(unique_images)}")

valid_dataset = []

for file in unique_images:

    label_file = file.rsplit(".", 1)[0] + ".txt"

    label_path = os.path.join(RAW_LABEL_DIR, label_file)

    if not os.path.exists(label_path):
        continue

    valid = True

    with open(label_path, "r") as f:

        lines = f.readlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:
            valid = False
            break

    if valid:
        valid_dataset.append(file)

print(f"Verified Labels: {len(valid_dataset)}")

helmet_count = 0
nohelmet_count = 0

for file in valid_dataset:

    label_file = file.rsplit(".", 1)[0] + ".txt"

    label_path = os.path.join(RAW_LABEL_DIR, label_file)

    with open(label_path, "r") as f:

        lines = f.readlines()

    for line in lines:

        class_id = int(line.split()[0])

        if class_id == 0:
            helmet_count += 1
        else:
            nohelmet_count += 1

print(f"Helmet Labels: {helmet_count}")
print(f"No Helmet Labels: {nohelmet_count}")

preprocessed_images = []

for file in valid_dataset:

    file_path = os.path.join(RAW_IMAGE_DIR, file)

    img = cv2.imread(file_path)

    img = cv2.resize(img, (640, 640))

    img = cv2.GaussianBlur(img, (3, 3), 0)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    cl = clahe.apply(l)

    enhanced = cv2.merge((cl, a, b))

    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    save_path = os.path.join(PROCESSED_DIR, file)

    cv2.imwrite(save_path, img)

    label_file = file.rsplit(".", 1)[0] + ".txt"

    src_label = os.path.join(RAW_LABEL_DIR, label_file)

    dst_label = os.path.join(PROCESSED_DIR, label_file)

    shutil.copy(src_label, dst_label)

    preprocessed_images.append(file)

print(f"Preprocessed Images: {len(preprocessed_images)}")

for file in preprocessed_images:

    file_path = os.path.join(PROCESSED_DIR, file)

    img = cv2.imread(file_path)

    filename = Path(file).stem

    flip = cv2.flip(img, 1)

    bright = cv2.convertScaleAbs(
        img,
        alpha=1.2,
        beta=25
    )

    h, w = img.shape[:2]

    matrix = cv2.getRotationMatrix2D(
        (w // 2, h // 2),
        15,
        1.0
    )

    rotated = cv2.warpAffine(
        img,
        matrix,
        (w, h)
    )

    cv2.imwrite(
        os.path.join(PROCESSED_DIR, f"{filename}_flip.jpg"),
        flip
    )

    cv2.imwrite(
        os.path.join(PROCESSED_DIR, f"{filename}_bright.jpg"),
        bright
    )

    cv2.imwrite(
        os.path.join(PROCESSED_DIR, f"{filename}_rot.jpg"),
        rotated
    )

print("Data Augmentation Completed")

hard_examples = []

for file in preprocessed_images:

    file_path = os.path.join(PROCESSED_DIR, file)

    img = cv2.imread(file_path)

    if img is None:
        continue

    h, w = img.shape[:2]

    if h < 300 or w < 300:
        hard_examples.append(file)

print(f"Hard Examples: {len(hard_examples)}")

dataset_images = []

for file in os.listdir(PROCESSED_DIR):

    if any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
        dataset_images.append(file)

print(dataset_images)
print(len(dataset_images))

if len(dataset_images) == 0:

    st.error("No images found in dataset")
    st.stop()

train_images, temp_images = train_test_split(
    dataset_images,
    test_size=0.30,
    random_state=42
)

val_images, test_images = train_test_split(
    temp_images,
    test_size=0.50,
    random_state=42
)

print(f"Train Images: {len(train_images)}")
print(f"Validation Images: {len(val_images)}")
print(f"Test Images: {len(test_images)}")

for split in ["train", "val", "test"]:

    os.makedirs(
        f"final_dataset/{split}/images",
        exist_ok=True
    )

    os.makedirs(
        f"final_dataset/{split}/labels",
        exist_ok=True
    )

def move_files(file_list, split):

    for file in file_list:

        src_img = os.path.join(PROCESSED_DIR, file)

        dst_img = os.path.join(
            f"final_dataset/{split}/images",
            file
        )

        shutil.copy(src_img, dst_img)

        label_file = file.rsplit(".", 1)[0] + ".txt"

        src_label = os.path.join(PROCESSED_DIR, label_file)

        if os.path.exists(src_label):

            dst_label = os.path.join(
                f"final_dataset/{split}/labels",
                label_file
            )

            shutil.copy(src_label, dst_label)

move_files(train_images, "train")
move_files(val_images, "val")
move_files(test_images, "test")

print("Dataset Organization Completed")

PERSON_MODEL = YOLO("yolov8n.pt")

HELMET_MODEL = YOLO(
    "runs/detect/train-5/weights/best.pt"
)

st.set_page_config(
    page_title="Helmet Detection",
    layout="centered"
)

st.title("🚦 Smart Helmet Detection System")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    img = np.array(image)

    annotated = img.copy()

    person_results = PERSON_MODEL(
        img,
        conf=0.4,
        verbose=False
    )

    helmet_results = HELMET_MODEL(
        img,
        conf=0.10,
        iou=0.45,
        verbose=False
    )

    helmets = []

    for r in helmet_results:

        for box in r.boxes:

            cls_id = int(box.cls[0])

            conf = float(box.conf[0])

            if conf < 0.10:
                continue

            if cls_id == 0:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                helmets.append(
                    (x1, y1, x2, y2)
                )

    def iou(boxA, boxB):

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter = max(
            0,
            xB - xA
        ) * max(
            0,
            yB - yA
        )

        if inter == 0:
            return 0

        areaA = (
            (boxA[2] - boxA[0]) *
            (boxA[3] - boxA[1])
        )

        return inter / areaA

    accepted = {
        "helmet": 0,
        "no-helmet": 0
    }

    for r in person_results:

        for box in r.boxes:

            if int(box.cls[0]) != 0:
                continue

            if float(box.conf[0]) < 0.4:
                continue

            px1, py1, px2, py2 = map(
                int,
                box.xyxy[0]
            )

            pw = px2 - px1
            ph = py2 - py1

            head_box = (
                px1 + int(pw * 0.15),
                py1,
                px2 - int(pw * 0.15),
                py1 + int(ph * 0.65)
            )

            padding = 10

            head_box = (
                head_box[0] - padding,
                head_box[1] - padding,
                head_box[2] + padding,
                head_box[3] + padding
            )

            has_helmet = False

            for h in helmets:

                if iou(head_box, h) > 0.08:
                    has_helmet = True
                    break

            if has_helmet:

                label = "Helmet"

                color = (0, 255, 0)

                accepted["helmet"] += 1

            else:

                label = "No Helmet"

                color = (0, 0, 255)

                accepted["no-helmet"] += 1

            cv2.rectangle(
                annotated,
                (px1, py1),
                (px2, py2),
                color,
                2
            )

            cv2.putText(
                annotated,
                label,
                (px1, py1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    st.image(
        annotated,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Helmet",
        accepted["helmet"]
    )

    col2.metric(
        "No Helmet",
        accepted["no-helmet"]
    )

    if accepted["no-helmet"] > 0:

        st.error(
            f"{accepted['no-helmet']} violation(s) detected!"
        )

    else:

        st.success(
            "All persons are wearing helmets"
        )