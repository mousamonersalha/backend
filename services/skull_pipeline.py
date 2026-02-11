import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 256
THRESH_BRAIN = 0.3
TUMOR_DILATION = 9


skull_model = tf.keras.models.load_model(
    "models/skull_stripping_model.keras",
    compile=False
)


# ----------------------------
# PREPROCESS
# ----------------------------
def skull_strip_preprocess(img):
    img = img.astype(np.float32)

    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip(img, p1, p99)

    img = (img - img.mean()) / (img.std() + 1e-8)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return img


# ----------------------------
# CLEAN MASK
# ----------------------------
def clean_brain_mask(mask):
    mask = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)

    h, w = mask.shape
    flood = mask.copy()
    flood_mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0,0), 1)

    mask = mask | (1 - flood)

    return mask


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def skull_strip(image):

    img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img_norm = skull_strip_preprocess(img_resized)
    img_input = np.expand_dims(img_norm, axis=0)

    pred_mask = skull_model.predict(img_input, verbose=0)[0]
    pred_mask = (pred_mask > THRESH_BRAIN).astype(np.uint8)
    pred_mask = pred_mask.squeeze()

    pred_mask = clean_brain_mask(pred_mask)

    brain_only = cv2.bitwise_and(
        img_resized,
        img_resized,
        mask=pred_mask * 255
    )

    return brain_only, pred_mask
