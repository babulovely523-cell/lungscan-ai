import numpy as np
import cv2
from PIL import Image
import os


def load_models():
    """
    Load trained models.
    Demo mode (no models) — random predictions chalengi.
    Jab aap apna model train karo, tab ye code uncomment karo:

        import tensorflow as tf
        cancer_model = tf.keras.models.load_model("models/cancer_model.h5")
        pneumonia_model = tf.keras.models.load_model("models/pneumonia_model.h5")
        return cancer_model, pneumonia_model
    """
    return None, None


def preprocess(img):
    """Convert PIL image to numpy array (224x224 normalized)."""
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, 0), arr


def predict(model, img_arr):
    """Return (result_str, confidence) — Detected / Not Detected."""
    if model is None:
        v = float(np.random.rand())
        return ("Detected" if v > 0.5 else "Not Detected"), v
    pred = model.predict(img_arr, verbose=0)[0][0]
    return ("Detected" if pred > 0.5 else "Not Detected"), float(pred)


def get_gradcam(model, img_array, last_conv_layer_name="out_relu"):
    """
    Return GradCAM heatmap (numpy array).
    Demo mode: random heatmap.
    """
    if model is None:
        h, w = img_array.shape[1], img_array.shape[2]
        heatmap = np.random.rand(h, w)
        return heatmap

    import tensorflow as tf
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        class_channel = preds[:, 0]
    grads = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_img, heatmap, alpha=0.45):
    """Return overlay image (numpy) with heatmap + bounding box."""
    img = np.array(original_img.convert("RGB").resize((224, 224)))
    hm = cv2.resize(heatmap, (224, 224))
    hm = np.uint8(255 * hm)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 1 - alpha, hm_color, alpha, 0)

    thr = hm > 150
    ys, xs = np.where(thr)
    if len(xs) > 0:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 3)
    return overlay
