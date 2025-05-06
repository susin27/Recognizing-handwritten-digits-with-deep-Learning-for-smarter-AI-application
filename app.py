import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64

app = Flask(__name__)
model = load_model("digit_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    image_data = base64.b64decode(data.split(',')[1])
    image = Image.open(io.BytesIO(image_data)).convert('L').resize((28, 28))
    image = 255 - np.array(image)  # Invert for black digits on white
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)

    prediction = model.predict(image)
    return jsonify({'prediction': int(np.argmax(prediction)), 'confidence': float(np.max(prediction))})

if __name__ == '__main__':
    app.run(debug=True)
