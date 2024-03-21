from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import pickle
# import tensorflow as tf

from utils.utils import resize_image, get_name, find_face


app = FastAPI()

model_path = './models/svm_model.pkl'

model_file = open(model_path, 'rb')
model = pickle.load(model_file)


@app.post('/identify_svm')
async def predict_svm(image: UploadFile = File(...)):
    image_bytes = image.file.read()

    image_array = cv2.imdecode(np.frombuffer(
        image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

    

    # find face
    found_face = find_face(image_array)
    
    if len(found_face) == 0:
        return {
            'prediction': 'No face detected'
        }
        
    # resize the image
    resized_image = resize_image(found_face)

    # prediction
    prediction = model.predict([resized_image.flatten()])

    # name
    name = get_name(idx=prediction.tolist()[0])
    print(f"prediction: {name}")
    return {
        'prediction': name
    }
