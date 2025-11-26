from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image
import io
import os
import uuid # Para generar nombres únicos para las imágenes procesadas
import numpy as np
import base64

import pandas as pd

# Cargar base de conocimiento
KNOWLEDGE_PATH = "conocimientos_fracturas.csv"

knowledge_df = pd.read_csv(KNOWLEDGE_PATH)

def search_knowledge(question: str):
    """Retorna el contenido más relevante según la pregunta"""
    question_lower = question.lower()
    best_match = None
    best_score = 0

    for _, row in knowledge_df.iterrows():
        score = 0
        for word in question_lower.split():
            if word in row["contenido"].lower() or word in row["titulo"].lower():
                score += 1

        if score > best_score:
            best_score = score
            best_match = row["contenido"]

    return best_match if best_match else "No se encontró información relevante."
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es")
qa_model = AutoModelForQuestionAnswering.from_pretrained("mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es")

def run_bert(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = qa_model(**inputs)

    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1

    answer = tokenizer.decode(inputs["input_ids"][0][start:end])
    return answer

# --- Configuración del Modelo ---
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"El modelo no se encontró en la ruta: {MODEL_PATH}. Copia el archivo best.pt aquí.")

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo YOLOv8: {e}")

app = FastAPI(title="API de Detección de Fracturas con YOLOv8")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoint para devolver imagen con detecciones ---
@app.post("/detect_fracture_image")
async def detect_fracture_base64(file: UploadFile = File(...), conf_threshold: float = 0.50):
    """
    Recibe una imagen, devuelve la imagen procesada con recuadros
    codificada en Base64, junto con las coordenadas de las detecciones.
    """
    
    # 1. Leer y convertir el archivo a objeto de imagen PIL
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo procesar la imagen subida.")

    # 2. Ejecutar la inferencia y obtener los resultados
    results = model.predict(
        source=image, 
        conf=conf_threshold, 
        save=False, 
        verbose=False, 
        imgsz=640
    )

    # 3. Obtener el array NumPy de la imagen con las cajas dibujadas
    im_np = results[0].plot()
    im_pil = Image.fromarray(im_np[..., ::-1]) 

    # 4. Guardar la imagen procesada en un buffer de memoria
    img_byte_arr = io.BytesIO()
    im_pil.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    # 5. Convertir el buffer de bytes a string Base64
    encoded_img = base64.b64encode(img_byte_arr.read())
    encoded_img_str = encoded_img.decode('utf-8')

    # 6. Extraer las coordenadas (detections) para mayor utilidad
    detections = results[0].boxes.data.tolist()
    formatted_detections = []
    for det in detections:
        formatted_detections.append({
            "box_2d": [int(coord) for coord in det[:4]],
            "confidence": det[4],
            "class_name": model.names[int(det[5])]
        })

    # 7. Devolver el JSON final
    return {
        "status": "success",
        "processed_image": {
            "mime_type": "image/jpeg",
            "base64_data": encoded_img_str
        },
        "detections": formatted_detections
    }
# --- Endpoint original para JSON (opcional, si aún lo necesitas) ---
# Puedes mantener este o eliminarlo, según tus necesidades.
@app.post("/detect_fracture_json")
async def detect_fracture_json(file: UploadFile = File(...), conf_threshold: float = 0.50):
    """
    Recibe una imagen y devuelve las detecciones de fracturas en formato JSON.
    """
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo procesar la imagen subida.")

    results = model.predict(source=image, conf=conf_threshold, save=False, verbose=False)
    
    detections = results[0].boxes.data.tolist()
    formatted_detections = []
    
    for det in detections:
        formatted_detections.append({
            "box_2d": [int(coord) for coord in det[:4]],
            "confidence": det[4],
            "class_id": int(det[5]),
            "class_name": model.names[int(det[5])]
        })

    return {
        "filename": file.filename,
        "detecciones_encontradas": len(formatted_detections),
        "detections": formatted_detections
    }

from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str

@app.post("/ask_assistant")
async def ask_assistant(payload: AskRequest):

    question = payload.question

    # Paso 1: buscar en la base de conocimiento
    context = search_knowledge(question)

    # Paso 2: obtener respuesta con BERT
    respuesta = run_bert(question, context)

    # Paso 3: fallback si BERT no produce respuesta
    if respuesta.strip() == "" or respuesta == "[CLS]":
        respuesta = context

    return {
        "pregunta": question,
        "respuesta": respuesta,
        "contexto_base": context
    }
