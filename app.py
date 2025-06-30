from flask import Flask, request, render_template, redirect
import os
import cv2
import numpy as np
import base64
from tensorflow.keras.models import load_model
from utils import predict_image_class

app = Flask(__name__)
model = load_model("Blood_Cell.h5")  # Make sure this file exists in the project root

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            file_path = os.path.join("static", file.filename)
            file.save(file_path)
            predicted_class_label, img_rgb = predict_image_class(file_path, model)
            _, img_encoded = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            img_str = base64.b64encode(img_encoded).decode('utf-8')
            return render_template("result.html", class_label=predicted_class_label, img_data=img_str)
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
