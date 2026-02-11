import tensorflow as tf
import numpy as np
import cv2
import os
import requests

IMG_SIZE = 224
SEG_THRESHOLD = 0.5

# =========================================
# Download Multitask Model from HuggingFace
# =========================================

MODEL_URL = "https://huggingface.co/mousasalha/multitask/resolve/main/best_multitask_stageB.keras"
MODEL_PATH = "best_multitask_stageB.keras"

if not os.path.exists(MODEL_PATH):
    print("Downloading Multitask Model from HuggingFace...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Multitask model downloaded.")

# =========================================
# Load Model
# =========================================

multitask_model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# ----------------------------
# PREPROCESS
# ----------------------------
def multitask_preprocess(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32)

    mean = gray.mean()
    std = gray.std() + 1e-8
    gray = (gray - mean) / std

    gray = np.expand_dims(gray, axis=-1)
    gray = np.repeat(gray, 3, axis=-1)
    gray = np.expand_dims(gray, axis=0)

    return gray


# ----------------------------
# PREDICTION
# ----------------------------
def multitask_predict(brain_image):

    processed = multitask_preprocess(brain_image)

    preds = multitask_model.predict(
    {"input_layer_5": processed},
    verbose=0
)


    segmentation_pred = preds[0]
    classification_pred = preds[1]

    class_id = int(np.argmax(classification_pred[0]))
    confidence = float(np.max(classification_pred[0]))

    seg_mask = segmentation_pred[0]
    seg_mask = (seg_mask > SEG_THRESHOLD).astype(np.uint8)
    seg_mask = seg_mask.squeeze()

    return class_id, confidence, seg_mask
