import tensorflow as tf
import numpy as np
import cv2


# =========================================================
# 🔥 Grad-CAM for Classification
# =========================================================
def generate_gradcam_classification(
    model,
    input_tensor,
    class_index,
    last_conv_layer_name
):
    """
    Generate Grad-CAM heatmap for classification head
    """

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output[1]  # classification head
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= tf.reduce_max(heatmap) + 1e-8


    return heatmap.numpy()


def generate_gradcam_segmentation_mask_guided(
    model,
    input_tensor,
    last_conv_layer_name,
    threshold=0.5
):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output[0]  # segmentation head
        ],
    )

    with tf.GradientTape() as tape:
        conv_out, seg_pred = grad_model(input_tensor)

        # 🔥 Create predicted mask
        predicted_mask = tf.cast(seg_pred > threshold, tf.float32)

        # 🔥 Mask-guided loss (focus only on tumor area)
        loss = tf.reduce_sum(seg_pred * predicted_mask)

    grads = tape.gradient(loss, conv_out)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_out = conv_out[0]

    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    heatmap = heatmap.numpy()

    h = input_tensor.shape[1]
    w = input_tensor.shape[2]

    heatmap = cv2.resize(heatmap, (w, h))

    return heatmap




# =========================================================
# 🔥 Grad-CAM for Segmentation
# =========================================================
def generate_gradcam_segmentation(
    model,
    input_tensor,
    last_conv_layer_name="conv2d_105"  # نفس النوتبوك
):

    seg_grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.get_layer("segmentation").output
        ]
    )

    with tf.GradientTape() as tape:
        conv_out, seg_pred = seg_grad_model(input_tensor)
        loss = tf.reduce_mean(seg_pred)

    grads = tape.gradient(loss, conv_out)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_out = conv_out[0]

    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    heatmap = heatmap.numpy()
    heatmap = cv2.resize(
        heatmap,
        (input_tensor.shape[2], input_tensor.shape[1])
    )

    return heatmap

# =========================================================
# 🎨 Convert Heatmap to Colored Image
# =========================================================
def apply_colormap_on_image(original_image, heatmap):
    """
    Overlay heatmap on original RGB image
    """

    heatmap_resized = cv2.resize(
        heatmap,
        (original_image.shape[1], original_image.shape[0])
    )

    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized),
        cv2.COLORMAP_JET
    )

    heatmap_colored = cv2.cvtColor(
        heatmap_colored,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original_image,
        0.6,
        heatmap_colored,
        0.4,
        0
    )

    return overlay
