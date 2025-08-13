from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import numpy as np
import earthaccess
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from datacube_generator import generate_prediction_datacube
from image_processor import convert_datacube_to_images
from joblib import load as joblib_load
from dependencies import get_api_key
from enum import Enum
from config import TIER_CONFIG
import uvicorn
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Flatten, Dense, Layer, Multiply, Permute, RepeatVector
import time 
from ultralytics import YOLO
import io
from fastapi.responses import JSONResponse
from image_drawer import draw_detections_on_image_and_save
from PIL import Image

# Initialize FastAPI
app = FastAPI(title="HAB Prediction API")

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.dense1 = Dense(1, activation='tanh')
        self.flatten = Flatten()
        self.softmax = Dense(10, activation='softmax')
        self.repeat = RepeatVector(256)
        self.permute = Permute([2, 1])

    def call(self, inputs):
        attention_scores = self.dense1(inputs)
        attention_weights = self.flatten(attention_scores)
        attention_weights = self.softmax(attention_weights)
        attention_weights = self.repeat(attention_weights)
        attention_weights = self.permute(attention_weights)
        return Multiply()([inputs, attention_weights])

models = {}
for tier, config in TIER_CONFIG.items():
    try:
        path = config["model_path"]
        if path.endswith(".pkl"):
            models[tier] = joblib_load(path)
        elif tier == "tier1":
            models[tier] = load_model(path)
        elif tier in ["tier2", "admin"]:
            models[tier] = load_model(path, custom_objects={"AttentionLayer": AttentionLayer})
        print(f"Loaded model for {tier} from {path}")
    except Exception as e:
        print(f"ERROR loading model for {tier}: {e}")
        models[tier] = None

# Load the YOLO model for object detection
yolo_model = YOLO("models/algae_detection_model.pt")

# Load environment variables
load_dotenv()
username = os.getenv("EARTHDATA_USERNAME")
password = os.getenv("EARTHDATA_PASSWORD")

try:
	earthaccess.login(strategy="environment")
except Exception as e:
	print(f"Error: NASA Earthdata login failed: {e}")

label_map = {0: "non-toxic", 1: "toxic"}

# This provides automatic data validation and documentation.
class Tier(str, Enum):
	FREE = "free"
	TIER_1 = "tier1"
	TIER_2 = "tier2"
	ADMIN = "admin"

class PredictionRequest(BaseModel):
	latitude: float
	longitude: float
	date: str
	tier: Tier

class ConfidenceScores(BaseModel):
	non_toxic: str
	toxic: str

class PredictionResponse(BaseModel):
	prediction_for_date: str
	predicted_label: str
	confidence_scores: ConfidenceScores

# API endpoint definition
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, api_key: str = Depends(get_api_key)):
	start = time.time()	

	# Select the config for the requested tier
	tier = request.tier.value
	config = TIER_CONFIG.get(tier)
	model = models.get(tier)
	model_name = os.path.basename(config['model_path'])
	print(f"[{tier.upper()}] Using model: {model_name}")

	if model is None:
		raise HTTPException(status_code=500, detail=f"Model for tier '{tier}' is not available or failed to load.")
		
	try:
		lat = request.latitude
		lon = request.longitude
		start_date_str = request.date

		# Generate the datacube with the specific config
		datacube = await run_in_threadpool(generate_prediction_datacube, lat, lon, start_date_str, config)

		if datacube is None:
			raise HTTPException(status_code=400, detail="Failed to generate datacube")
		
		# Convert to resized, normalized, concatenated image sequence
		# The output shape is now (10, 64, 64, 9)
		image_sequence = await run_in_threadpool(convert_datacube_to_images, datacube, config, tier)

		'''
		# Prepare sequence for model
		# Current shape is 4D: (time_steps, height, width, channels)-> (10, 64, 64, 9), we need to add the batch dimension: (batch_size, time_steps, height, width, channels) for the model -> (1, 10, 64, 64, 9)
		# batch: How many datacubes we are predicting at once
		'''
		# # Adding batch dimension - (1, 10, 64, 64, 9)
		# model_input = np.expand_dims(image_sequence, axis=0)
			
		# The training code scales by 255.0, so we do the same here
		# model_input = model_input.astype(np.float32) / 255.0
		# Flatten the (10, 64, 64, 9) array into a 1D vector
		prediction_probs = []

		if tier == 'free':
			flattened_features = image_sequence.flatten()
			model_input = flattened_features.reshape(1, -1)
			prediction_probs = model.predict_proba(model_input)[0]
			
		elif tier in ['tier1', 'tier2', 'admin']:
			print(f'Into {tier} TIER ::::::::')
			model_input = np.expand_dims(image_sequence,0)
			print(f"Final model input shape: {model_input.shape}, dtype: {model_input.dtype}")
			prediction_probs = model.predict(model_input)[0]
		
		print(prediction_probs, int(time.time() - start))
		# Extract the probabilities for non-toxic and toxic
		prob_non_toxic = prediction_probs[0]
		prob_toxic = prediction_probs[1]

		# Determine the predicted label
		predicted_class_index = np.argmax(prediction_probs)
		predicted_label = label_map[predicted_class_index]

		# Calculate the date that was predicted for (start_date + 5 days)
		prediction_target_date = datetime.strptime(start_date_str, '%Y-%m-%d') + timedelta(days=config['days'])

		# Return a Pydantic model, which FastAPI automatically converts to JSON.
		return PredictionResponse(
			prediction_for_date=prediction_target_date.strftime('%Y-%m-%d'),
			predicted_label=predicted_label,
			confidence_scores=ConfidenceScores(
					non_toxic=f"{prob_non_toxic:.4f}",
					toxic=f"{prob_toxic:.4f}"
			)
		)

	except Exception as e:
		print(f"ERROR during prediction: {e}")
		raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

@app.post("/predictimage")
async def predictimage(file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    print("Into OBJECT DETECTION CODE :::::::")
    try:
    # Check file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Unsupported file type.")
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        results = yolo_model(image, imgsz = 640)
        detections = []
        for box in results[0].boxes:
            print(f"CONFIDENCE ::::::::::::::::::::::: {float(box.conf)}")
            if float(box.conf) > 0.6:
                detections.append({
                "class": int(box.cls),
                "label": results[0].names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist()
            })
            
            
        print('YOLO DETECTION COMPLETED :::::::::::::::::')
        processed_img_b64 = draw_detections_on_image_and_save(image, detections)
        print('BOUNDING BOX CREATION COMPLETED :::::::::::::::::')
        if not detections:
            print('NOTHING DETECTED :::::::::::::::::')
            return JSONResponse(content={"output_image_url": processed_img_b64,"label": "No Algae Detected"})
        return JSONResponse(content={"output_image_url": processed_img_b64, "label":"Algae Detected"})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

	
if __name__ == '__main__':
	port = int(os.environ.get("PORT", 8080))
	uvicorn.run(app, host="0.0.0.0", port=port)