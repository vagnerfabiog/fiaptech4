from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from pydantic import BaseModel
import numpy as np
from keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler
import os,json
import time
from prometheus_fastapi_instrumentator import Instrumentator
import logging
from typing import List

# Configuração básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregar artefatos do modelo
model_dir = 'model_artifacts'
model = load_model(os.path.join(model_dir, 'lstm_model.h5'))
scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))

with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
    metadata = json.load(f)

look_back = metadata['look_back']

app = FastAPI(title="Stock Price Prediction API",
              description="API para previsão de preços de ações usando LSTM",
              version="1.0")

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de requisição
class PredictionRequest(BaseModel):
    historical_prices: List[float]
    future_steps: int = 1  # Número de passos futuros para prever

# Modelo de resposta
class PredictionResponse(BaseModel):
    predictions: List[float]
    processing_time: float
    model_metrics: dict

# Instrumentação Prometheus
Instrumentator().instrument(app).expose(app)

# Configuração para servir arquivos estáticos e templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health", tags=["Monitoring"])
async def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    start_time = time.time()
    
    try:
        # Verificar se temos dados suficientes
        if len(request.historical_prices) < look_back:
            raise HTTPException(
                status_code=400,
                detail=f"Necessário pelo menos {look_back} pontos históricos para previsão"
            )
        
        # Preparar dados
        historical_data = np.array(request.historical_prices[-look_back:]).reshape(-1, 1)
        scaled_data = scaler.transform(historical_data)
        
        # Fazer previsões
        predictions = []
        current_batch = scaled_data.reshape(1, look_back, 1)
        
        for _ in range(request.future_steps):
            current_pred = model.predict(current_batch)[0]
            predictions.append(float(current_pred))
            
            # Atualizar batch para incluir a previsão
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
        
        # Reverter a normalização
        predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten().tolist()
        
        processing_time = time.time() - start_time
        
        logger.info(f"Prediction completed in {processing_time:.2f} seconds")
        
        return PredictionResponse(
            predictions=predicted_prices,
            processing_time=processing_time,
            model_metrics=metadata['metrics']
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))