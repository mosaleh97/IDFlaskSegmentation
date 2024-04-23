from ultralytics import YOLO

import cv2
from utils.Transformations import convert_object


def predict_mask(image, model):
    H, W, _ = image.shape
    result = model(image)[0]

    mask = result.masks.data[0].cpu().numpy() * 255

    return cv2.resize(mask, (W, H))


def get_wrapped_card(image, mask):
    return convert_object(mask, image)


'''
model_path = "../models/last.pt"

image_path = "C:/Users/alex/Downloads/WhatsApp Image 2024-03-30 at 07.02.01_30d53384.jpg"

img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)

results = model(img)

for result in results:
    for j, mask in enumerate(result.masks.data):
        # print unique values of mask
        print(set(mask.cpu().numpy().flatten()))

        mask = mask.cpu().numpy() * 255

        mask = cv2.resize(mask, (W, H))

        cv2.imwrite('../outputresized.png', mask)
'''
