import tensorflow as tf
import numpy as np
import cv2

IMG_SIZE = 224
SEG_THRESHOLD = 0.5

# Load Stage B model
multitask_model = tf.keras.models.load_model(
    "models/best_multitask_stageB.keras",
    compile=False
)

# ----------------------------
# PREPROCESS
# ----------------------------
def multitask_preprocess(image):

    # Convert to grayscale (important!)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize to 224
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    gray = gray.astype(np.float32)

    # Medical Z-score normalization (per image)
    mean = gray.mean()
    std = gray.std() + 1e-8
    gray = (gray - mean) / std

    # Expand dims to match model input
    gray = np.expand_dims(gray, axis=-1)  # (224,224,1)
    gray = np.repeat(gray, 3, axis=-1)    # convert to (224,224,3) if model expects 3 channels

    gray = np.expand_dims(gray, axis=0)   # batch dimension

    return gray



# ----------------------------
# PREDICTION
# ----------------------------
def multitask_predict(brain_image):

    processed = multitask_preprocess(brain_image)

    preds = multitask_model.predict(processed, verbose=0)

    # Since predict returns list
    segmentation_pred = preds[0]
    classification_pred = preds[1]

    # Classification
    class_id = int(np.argmax(classification_pred[0]))
    confidence = float(np.max(classification_pred[0]))

    # Segmentation
    seg_mask = segmentation_pred[0]
    seg_mask = (seg_mask > 0.5).astype(np.uint8)
    seg_mask = seg_mask.squeeze()

    return class_id, confidence, seg_mask

