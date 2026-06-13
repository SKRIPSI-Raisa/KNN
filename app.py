import os
import cv2
import joblib
import numpy as np
import time
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from skimage.feature import graycomatrix, graycoprops

# Configuration
IMAGE_SIZE = (128, 128)

# Global variables for models
model = None
scaler = None
le = None

def extract_rgb_features(image_rgb):
    """
    Extracts mean and standard deviation from R, G, B channels.
    Matches the implementation in training notebooks and camera.py.
    """
    features = []
    for c in range(3):
        channel = image_rgb[:, :, c]
        features.append(float(np.mean(channel)))
        features.append(float(np.std(channel)))
    return features

def extract_glcm_features(gray_image):
    """
    Extracts GLCM texture features (Contrast, Energy, Homogeneity, Correlation).
    Matches the implementation in training notebooks and camera.py.
    """
    glcm = graycomatrix(
        gray_image,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )
    contrast = float(graycoprops(glcm, "contrast").mean())
    energy = float(graycoprops(glcm, "energy").mean())
    homogeneity = float(graycoprops(glcm, "homogeneity").mean())
    correlation = float(graycoprops(glcm, "correlation").mean())
    return [contrast, energy, homogeneity, correlation]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, le
    
    # Resolve absolute paths relative to app.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "knn_best.pkl")
    scaler_path = os.path.join(current_dir, "scaler.pkl")
    le_path = os.path.join(current_dir, "label_encoder.pkl")
    
    print("=" * 60)
    print("Initializing KNN Waste Classifier API...")
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(le_path)):
        error_msg = (
            f"Model files not found. Ensure the following files exist in {current_dir}:\n"
            f" - knn_best.pkl\n"
            f" - scaler.pkl\n"
            f" - label_encoder.pkl"
        )
        print(f"[ERROR] {error_msg}")
        raise RuntimeError(error_msg)
        
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        le = joblib.load(le_path)
        print("Model, Scaler, and Label Encoder successfully loaded!")
        print(f"Registered Classes: {list(le.classes_)}")
        print(f"Model Neighbors (K): {model.n_neighbors}")
        print("=" * 60)
    except Exception as e:
        print(f"[ERROR] Failed to load model files: {e}")
        raise RuntimeError(f"Model initialization failed: {e}")
        
    yield
    
    # Cleanup resources (if any)
    print("Shutting down API...")

# Initialize FastAPI App
app = FastAPI(
    title="WasteSort KNN API (GLCM + RGB)",
    description="FastAPI backend providing waste classification (Organik / Non-Organik) using KNN with RGB color and GLCM texture features.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    model_loaded = model is not None and scaler is not None and le is not None
    return {
        "app": "WasteSort KNN API (GLCM + RGB)",
        "status": "Online",
        "model_status": "Loaded" if model_loaded else "Not Loaded",
        "k_value": model.n_neighbors if model_loaded else None,
        "classes": list(le.classes_) if model_loaded else [],
        "features_supported": [
            "RGB Mean and Standard Deviation (6 features)",
            "GLCM Texture Metrics: Contrast, Energy, Homogeneity, Correlation (4 features)"
        ]
    }

@app.post("/api/classify")
async def classify_image(file: UploadFile = File(...)):
    global model, scaler, le
    
    if model is None or scaler is None or le is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please verify the server startup logs."
        )
        
    start_time = time.time()
    
    try:
        # 1. Read uploaded image bytes
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid or readable image."
            )
            
        # 2. Preprocess: Resize image to training size
        img_resized = cv2.resize(img, IMAGE_SIZE)
        rgb_image = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # 3. Feature extraction
        rgb_feats = extract_rgb_features(rgb_image)
        glcm_feats = extract_glcm_features(gray_image)
        combined_features = rgb_feats + glcm_feats
        
        # 4. Scale features
        features_scaled = scaler.transform([combined_features])
        
        # 5. Prediction
        pred_class_idx = int(model.predict(features_scaled)[0])
        pred_class = str(le.inverse_transform([pred_class_idx])[0])
        
        # 6. Confidence Score / Probabilities
        try:
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = float(probabilities[pred_class_idx] * 100)
        except Exception:
            confidence = 100.0  # Fallback if predict_proba is not supported or fails
            
        # 7. Get Nearest Neighbors Details (Distances & Labels)
        neighbors = []
        try:
            distances, indices = model.kneighbors(features_scaled)
            distances = distances[0]
            indices = indices[0]
            
            for rank_idx, (idx, dist) in enumerate(zip(indices, distances)):
                # Reconstruct neighbor class label using model._y
                neighbor_class_idx = model._y[idx]
                neighbor_label = str(le.inverse_transform([neighbor_class_idx])[0])
                
                neighbors.append({
                    "rank": rank_idx + 1,
                    "distance": float(round(dist, 4)),
                    "label": neighbor_label
                })
        except Exception as e:
            print(f"[WARNING] Could not retrieve nearest neighbors: {e}")
            
        execution_time = time.time() - start_time
        
        # 8. Encode processed image (128x128) to base64 for preview
        _, buffer = cv2.imencode('.jpg', img_resized)
        img_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        
        return {
            "status": "success",
            "filename": file.filename,
            "prediction": pred_label_mapping(pred_class),
            "confidence_percent": float(round(confidence, 2)),
            "features": {
                "rgb_mean_std": {
                    "red": {"mean": rgb_feats[0], "std": rgb_feats[1]},
                    "green": {"mean": rgb_feats[2], "std": rgb_feats[3]},
                    "blue": {"mean": rgb_feats[4], "std": rgb_feats[5]}
                },
                "glcm_texture": {
                    "contrast": glcm_feats[0],
                    "energy": glcm_feats[1],
                    "homogeneity": glcm_feats[2],
                    "correlation": glcm_feats[3]
                }
            },
            "k_neighbors_count": len(neighbors),
            "neighbors": neighbors,
            "processed_image_url": img_base64,
            "execution_time_seconds": float(round(execution_time, 4))
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error occurred: {str(e)}"
        )

def pred_label_mapping(label: str) -> str:
    """
    Standardizes labels to human-readable format.
    E.g., "organik" -> "Organik", "non_organik" -> "Non-Organik"
    """
    cleaned = label.lower().strip()
    if cleaned == "organik":
        return "Organik"
    elif cleaned in ["non_organik", "non-organik", "nonorganik"]:
        return "Non-Organik"
    return label.capitalize()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
