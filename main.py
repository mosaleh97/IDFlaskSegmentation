from flask import Flask, request, Response, jsonify
import json
from inference.get_card import predict_mask, get_wrapped_card
import cv2
import numpy as np
import io
from ultralytics import YOLO

app = Flask(__name__)

# Read json file having the configuration
with open('config/model_config.json') as f:
    config = json.load(f)

# Load YOLOv8s model
model = YOLO(config['model_path'])  # Load YOLOv8s model


# Define a post request endpoint that receives the image and returns the wrapped card
@app.route('/get_id_card', methods=['POST'])
def get_id_card():
    # Services ------
    # Get the image file
    file = request.files['image']

    # Read the image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Get the mask of the image
    mask = predict_mask(img, model)

    # Write the mask
    cv2.imwrite('outputresized.png', mask)

    wrapped_card = get_wrapped_card(img, mask)

    if wrapped_card is None:
        error_msg = {"detail": config['error_message']}
        return jsonify(error_msg), 422

    # Write the wrapped card
    cv2.imwrite('outputwrapped.jpeg', wrapped_card)

    # Convert image to bytes
    _, img_encoded = cv2.imencode('.jpeg', wrapped_card)
    img_bytes = img_encoded.tobytes()

    # Return the image as a response
    response = Response(io.BytesIO(img_bytes))
    response.headers['Content-Type'] = 'image/jpeg'
    return response


if __name__ == '__main__':
    app.run(host="localhost", port=8000)
