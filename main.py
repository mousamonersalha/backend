from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
import base64
import tensorflow as tf

from services.skull_pipeline import skull_strip
from services.multitask_pipeline import (
    multitask_predict,
    multitask_preprocess,
    multitask_model
)

from utils.gradcam_utils import (
    generate_gradcam_classification,
    generate_gradcam_segmentation,
    apply_colormap_on_image
)

app = FastAPI()

LAST_CONV_LAYER = "conv2d_99"  # تأكد الاسم صحيح


# =========================================================
# 🔄 Convert image to Base64
# =========================================================
def image_to_base64(image):
    _, buffer = cv2.imencode(".png", image)
    encoded = base64.b64encode(buffer).decode("utf-8")
    return encoded


# =========================================================
# 🧠 Skull Strip Endpoint
# =========================================================
@app.post("/skull-strip")
async def skull_strip_endpoint(file: UploadFile = File(...)):

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    brain_only, mask = skull_strip(image)

    return {
        "brain_image": image_to_base64(brain_only),
        "brain_mask": image_to_base64(mask * 255)
    }


# =========================================================
# 🧠 Full Pipeline Endpoint
# =========================================================
@app.post("/predict")
async def full_pipeline(file: UploadFile = File(...)):

    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # -------------------------------------------------
        # 1️⃣ Skull Stripping
        # -------------------------------------------------
        brain_only, brain_mask = skull_strip(image)

        # -------------------------------------------------
        # 2️⃣ Multi-Task Prediction
        # -------------------------------------------------
        class_id, confidence, tumor_mask = multitask_predict(brain_only)

        # Resize tumor mask back to original size
        tumor_mask_resized = cv2.resize(
            tumor_mask,
            (brain_only.shape[1], brain_only.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # Overlay tumor on brain
        overlay = brain_only.copy()
        overlay[tumor_mask_resized == 1] = [255, 0, 0]

        # Tumor percentage
        # Tumor percentage (مقارنة بحجم الدماغ فقط)

        tumor_pixels = np.sum(tumor_mask)

        # resize brain mask لنفس حجم tumor_mask
        brain_mask_resized = cv2.resize(
            brain_mask,
            (tumor_mask.shape[1], tumor_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        brain_pixels = np.sum(brain_mask_resized)

        tumor_percentage = float(
            (tumor_pixels / (brain_pixels + 1e-8)) * 100
        )


        # -------------------------------------------------
        # 3️⃣ Grad-CAM
        # -------------------------------------------------
        processed_tensor = multitask_preprocess(brain_only)

        heatmap_class = generate_gradcam_classification(
            multitask_model,
            processed_tensor,
            class_id,
            LAST_CONV_LAYER
        )

        heatmap_seg = generate_gradcam_segmentation(
            multitask_model,
            processed_tensor,
            LAST_CONV_LAYER
        )

        overlay_class, _ = apply_colormap_on_image(brain_only, heatmap_class)
        overlay_seg, _ = apply_colormap_on_image(brain_only, heatmap_seg)

        # -------------------------------------------------
        # 4️⃣ Return Everything
        # -------------------------------------------------
        return {
            "class_id": class_id,
            "confidence": confidence,
            "tumor_percentage": tumor_percentage,

            "brain_image": image_to_base64(brain_only),
            "brain_mask": image_to_base64(brain_mask * 255),
            "tumor_mask": image_to_base64(tumor_mask_resized * 255),
            "overlay": image_to_base64(overlay),

            "gradcam_classification": image_to_base64(overlay_class),
            "gradcam_segmentation": image_to_base64(overlay_seg)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
