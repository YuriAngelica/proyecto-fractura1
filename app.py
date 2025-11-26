from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API de Detección de Fracturas Óseas")

# Configurar CORS para permitir peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
MODEL = None
MODEL_PATH = "modelo.h5"  # Ruta a tu modelo .h5
IMG_SIZE = (224, 224)  # Ajusta según tu modelo

@app.on_event("startup")
async def load_model_on_startup():
    """Cargar el modelo al iniciar la aplicación"""
    global MODEL
    try:
        logger.info(f"Cargando modelo desde {MODEL_PATH}...")
        MODEL = load_model(MODEL_PATH)
        logger.info("Modelo cargado exitosamente")
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {str(e)}")
        raise

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesar la imagen para el modelo
    Ajusta según los requisitos de tu modelo
    """
    # Convertir a RGB si es necesario
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionar
    image = image.resize(IMG_SIZE)
    
    # Convertir a array numpy
    img_array = np.array(image)
    
    # Normalizar (ajusta según tu modelo)
    img_array = img_array / 255.0
    
    # Añadir dimensión del batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.get("/")
async def root():
    """Endpoint raíz para verificar que el API está funcionando"""
    return {
        "message": "API de Detección de Fracturas Óseas",
        "status": "activo",
        "modelo_cargado": MODEL is not None
    }

@app.get("/health")
async def health_check():
    """Verificar el estado del servicio"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return {"status": "healthy", "modelo": "cargado"}

@app.post("/predict")
async def predict_fracture(file: UploadFile = File(...)):
    """
    Endpoint para predecir si hay fractura en una radiografía
    
    Args:
        file: Imagen de radiografía (JPG, PNG)
    
    Returns:
        JSON con la predicción y nivel de confianza
    """
    # Validar que el modelo esté cargado
    if MODEL is None:
        raise HTTPException(
            status_code=503, 
            detail="Modelo no disponible. El servidor está iniciando."
        )
    
    # Validar tipo de archivo
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen (JPG, PNG)"
        )
    
    try:
        # Leer la imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        logger.info(f"Procesando imagen: {file.filename}, tamaño: {image.size}")
        
        # Preprocesar la imagen
        processed_image = preprocess_image(image)
        
        # Realizar predicción
        prediction = MODEL.predict(processed_image)
        logger.error(prediction)
        # Interpretar resultados
        # Ajusta según la salida de tu modelo
        # Asumiendo salida binaria: [probabilidad_normal, probabilidad_fractura]
        confidence = float(prediction[0][0])
        
        # Si tu modelo retorna una única probabilidad (fractura)
        is_fractured = confidence > 0.5
        
        result = {
            "prediction": "fractured" if is_fractured else "normal",
            "confidence": confidence if is_fractured else (1 - confidence),
            "raw_prediction": float(confidence),
            "filename": file.filename
        }
        
        logger.info(f"Resultado: {result['prediction']} (confianza: {result['confidence']:.2%})")
        
        return result
        
    except Exception as e:
        logger.error(f"Error al procesar imagen: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la imagen: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)