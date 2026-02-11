import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# ============================
# Load image
# ============================
IMAGE_PATH = r"C:\Users\mousa\Desktop\backend\Tr-gl_0038.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image not found!")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ============================
# 1️⃣ Skull Stripping
# ============================
brain_only, brain_mask = skull_strip(image)

# ============================
# 2️⃣ MultiTask Stage B
# ============================
class_id, confidence, tumor_mask = multitask_predict(brain_only)

# Resize tumor mask to brain size
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


# ============================
# 🧮 Tumor Percentage
# نسبة الورم مقارنة بحجم الدماغ فقط
# ============================
tumor_pixels = np.sum(tumor_mask)
brain_pixels = np.sum(brain_mask)

tumor_percentage = float(
    (tumor_pixels / (brain_pixels + 1e-8)) * 100
)

# ============================
# 3️⃣ Grad-CAM
# ============================
LAST_CONV_LAYER = "conv2d_99"  # تأكد الاسم صحيح

processed_tensor = multitask_preprocess(brain_only)

# Classification GradCAM
heatmap_class = generate_gradcam_classification(
    multitask_model,
    processed_tensor,
    class_id,
    LAST_CONV_LAYER
)

# Segmentation GradCAM
heatmap_seg = generate_gradcam_segmentation(
    multitask_model,
    processed_tensor
)

# Apply overlays (🔥 مهم نفك tuple)
overlay_class, _ = apply_colormap_on_image(brain_only, heatmap_class)
overlay_seg, _ = apply_colormap_on_image(brain_only, heatmap_seg)

# ============================
# 4️⃣ Visualization
# ============================
plt.figure(figsize=(18, 12))

plt.subplot(3, 3, 1)
plt.title("Original")
plt.imshow(image)
plt.axis("off")

plt.subplot(3, 3, 2)
plt.title("Skull Mask")
plt.imshow(brain_mask, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 3)
plt.title("Brain Only")
plt.imshow(brain_only)
plt.axis("off")

plt.subplot(3, 3, 4)
plt.title("Tumor Mask")
plt.imshow(tumor_mask_resized, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 5)
plt.title("Tumor Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.subplot(3, 3, 6)
plt.title("Grad-CAM Classification")
plt.imshow(overlay_class)
plt.axis("off")

plt.subplot(3, 3, 7)
plt.title("Grad-CAM Segmentation")
plt.imshow(overlay_seg)
plt.axis("off")

plt.subplot(3, 3, 8)
plt.text(
    0.05,
    0.5,
    f"Class ID: {class_id}\n\n"
    f"Confidence: {confidence:.4f}\n\n"
    f"Tumor % (Brain): {tumor_percentage:.2f}%",
    fontsize=14
)
plt.axis("off")

plt.tight_layout()
plt.show()
