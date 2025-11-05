# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
import re
import io
import traceback

app = Flask(__name__)

# --- Load model (helpful error message if missing/broken) ---
MODEL_PATH = "digit_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"Loaded model: {MODEL_PATH}")
except Exception as e:
    print("ERROR loading model:", e)
    model = None

# --- Image preprocessing helper ---
def preprocess_base64_image(b64_string):
    """
    Input: base64 data URL (e.g. "data:image/png;base64,...")
    Output: np array shaped (1,28,28,1), dtype float32 scaled [0,1]
    Steps:
      - decode base64
      - open with PIL, convert to L (grayscale)
      - detect background and invert if required to match MNIST (white digit on black)
      - crop bounding box of the digit, resize to ~20x20, paste centered on 28x28
      - contrast stretch if needed
    """
    # remove header if present
    b64_data = re.sub(r'^data:image/.+;base64,', '', b64_string)
    image_bytes = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale

    # convert to numpy array
    arr = np.array(img).astype(np.uint8)

    # If the canvas background is white (mean high), invert so digit becomes white on black
    if arr.mean() > 127:
        arr = 255 - arr

    # simple contrast stretch (avoid flat images)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = ((arr - mn) * (255.0 / (mx - mn))).clip(0, 255).astype(np.uint8)

    # find bounding box of non-zero (digit) pixels
    mask = arr > 20  # threshold to find digit pixels (tunable)
    coords = np.column_stack(np.where(mask))
    if coords.size:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        cropped = Image.fromarray(arr[y0:y1 + 1, x0:x1 + 1])
        # resize the digit region to fit into ~20x20 keeping aspect ratio
        cropped.thumbnail((20, 20), Image.LANCZOS)
        # create new 28x28 image and paste centered
        new_img = Image.new('L', (28, 28), color=0)  # black background
        left = (28 - cropped.size[0]) // 2
        top = (28 - cropped.size[1]) // 2
        new_img.paste(cropped, (left, top))
        final_arr = np.array(new_img)
    else:
        # fallback: resize the full image
        final_img = Image.fromarray(arr).resize((28, 28), Image.LANCZOS)
        final_arr = np.array(final_img)

    # normalize to 0-1 (as float32). MNIST expects white-on-black digit, so final_arr already has that.
    final_arr = final_arr.astype("float32") / 255.0
    final_arr = final_arr.reshape(1, 28, 28, 1)
    return final_arr

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Train or place digit_model.h5 next to app.py."}), 500

    try:
        payload = request.get_json(force=True)
        if not payload or "image" not in payload:
            return jsonify({"error": "No image provided in JSON body under key 'image'"}), 400

        b64_image = payload["image"]
        img_arr = preprocess_base64_image(b64_image)

        # model prediction
        preds = model.predict(img_arr)  # shape (1,10)
        probs = preds[0]
        top_idx = probs.argsort()[-3:][::-1]  # top 3 indices

        top3 = [
            {"digit": int(i), "probability": float(round(probs[i], 4))}
            for i in top_idx
        ]

        return jsonify({
            "digit": int(top_idx[0]),
            "predictions": top3
        })
    except Exception as e:
        print("Exception in /predict:", e)
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
