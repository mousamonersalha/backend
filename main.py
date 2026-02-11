from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
import tensorflow as tf
import os
import traceback

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

# =========================================================
# 🚀 App Init
# =========================================================
app = FastAPI()

# 🔥Flutter Web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LAST_CONV_LAYER = "conv2d_99"


# =========================================================
# 🔄 Convert image to Base64
# =========================================================
def image_to_base64(image):
    if image is None:
        return None

    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # لو float نحولها uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")



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
        "brain_mask": image_to_base64(mask.astype(np.uint8) * 255)
    }


# =========================================================
# 🧠 Full AI Pipeline
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

        # =================================================
        # 1️⃣ Skull Stripping
        # =================================================
        brain_only, brain_mask = skull_strip(image)

        # =================================================
        # 2️⃣ Multi-Task Prediction
        # =================================================
        class_id, confidence, tumor_mask = multitask_predict(brain_only)

        # Resize tumor mask back to original size
        tumor_mask_resized = cv2.resize(
            tumor_mask,
            (brain_only.shape[1], brain_only.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # Overlay tumor
        overlay = brain_only.copy()

        # Create red layer
        red_layer = np.zeros_like(brain_only)
        red_layer[:, :, 0] = 255  # Red channel

        alpha = 0.3  # 👈 
        # Apply blending only on tumor region
        mask_3d = np.repeat(tumor_mask_resized[:, :, np.newaxis], 3, axis=2)

        overlay = np.where(
            mask_3d == 1,
            cv2.addWeighted(brain_only, 1 - alpha, red_layer, alpha, 0),
            brain_only
        )


        # =================================================
        # 3️⃣ Tumor Percentage (مقارنة بحجم الدماغ)
        # =================================================
        tumor_pixels = np.sum(tumor_mask)

        brain_mask_resized = cv2.resize(
            brain_mask,
            (tumor_mask.shape[1], tumor_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        brain_pixels = np.sum(brain_mask_resized)

        tumor_percentage = float(
            (tumor_pixels / (brain_pixels + 1e-8)) * 100
        )

        # =================================================
        # 4️⃣ Grad-CAM
        # =================================================
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

        overlay_class = apply_colormap_on_image(brain_only, heatmap_class)
        overlay_seg = apply_colormap_on_image(brain_only, heatmap_seg)

        # =================================================
        # 5️⃣ Return Response
        # =================================================
        return {
            "class_id": class_id,
            "confidence": confidence,
            "tumor_percentage": tumor_percentage,

            "brain_image": image_to_base64(brain_only),
            "brain_mask": image_to_base64(brain_mask.astype(np.uint8) * 255),
            "tumor_mask": image_to_base64(tumor_mask_resized.astype(np.uint8) * 255),
            "tumor_overlay": image_to_base64(overlay),

            "gradcam_classification": image_to_base64(overlay_class),
            "gradcam_segmentation": image_to_base64(overlay_seg)
        }


    except Exception as e:
        print("🔥 FULL ERROR:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))



# =========================================================
# ▶ Run Server
# =========================================================
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port
    )
