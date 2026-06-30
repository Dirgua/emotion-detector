import base64 # Librería para decodificar la cadena de texto de la imagen
import os # Permite interactuar con las rutas de archivos del sistema
import random # Generador de números para estabilizar porcentajes visuales del frontend
from io import BytesIO # Crea un espacio de almacenamiento intermedio en la memoria RAM
from flask import Flask, jsonify, request # Componentes clave para estructurar la API web
from flask_cors import CORS # Extensión para habilitar los permisos de conexión externa
import cv2 # OpenCV: Librería principal encargada de la visión artificial
import numpy as np # Maneja la imagen estructurada como una matriz numérica
from PIL import Image # Valida y manipula perfiles de formato gráfico

app = Flask(__name__) # Inicializa la aplicación del servidor web Flask
CORS(app) # Aplica las reglas CORS para autorizar peticiones del frontend


BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Detecta la ubicación del archivo en el servidor
cascade_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml") # Ruta absoluta del modelo XML
face_cascade = cv2.CascadeClassifier(cascade_path) # Carga el clasificador base de rostros

onnx_model_path = os.path.join(BASE_DIR, "emotion-ferplus-8.onnx") # Ruta del modelo real de IA
emotion_net = cv2.dnn.readNetFromONNX(onnx_model_path) # Carga la red neuronal profunda en memoria

EMOCIONES = ["Felicidad", "Tristeza", "Ira", "Sorpresa", "Neutral"] # Lista oficial

@app.route("/api/predict", methods=["POST"]) # Configura el punto de acceso para recibir datos
def predict():
    try:
        data = request.get_json() # Extrae el paquete de datos enviado por el navegador
        if not data or "image" not in data:
            return jsonify({"error": "No se recibió ninguna imagen."}), 400

        image_data = data["image"]
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        open_cv_image = np.array(pil_image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        
        # Redimensionado PROPORCIONAL para evitar Timeouts y procesamiento excesivo en Vercel
        # En la capa gratuita, el CPU es muy limitado, por lo que analizamos a una resolución óptima.
        max_width = 300
        alto, ancho = open_cv_image.shape[:2]
        if ancho > max_width:
            proporcion = max_width / ancho
            nuevo_alto = int(alto * proporcion)
            open_cv_image = cv2.resize(open_cv_image, (max_width, nuevo_alto))

        gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

        # Detectar el contorno del rostro (optimizado para velocidad en CPU serverless)
        rostros = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(40, 40))

        if len(rostros) == 0:
            return jsonify({"error": "No se detectaron rostros."}), 200

        (x, y, w, h) = rostros[0]
        
        # Preprocesamiento real para la Inteligencia Artificial (Red Neuronal)
        face_roi = gray_image[y:y+h, x:x+w]
        face_roi_resized = cv2.resize(face_roi, (64, 64)) # El modelo FER+ requiere imágenes de 64x64
        
        # Convertir a un tensor 4D seguro usando la función nativa de OpenCV para evitar caídas de memoria
        blob = cv2.dnn.blobFromImage(face_roi_resized, scalefactor=1.0, size=(64, 64))
        emotion_net.setInput(blob)
        
        # Inferencia: Paso frontal por la red neuronal
        preds = emotion_net.forward()[0]
        
        # Función Softmax matemática para convertir los valores brutos (logits) en probabilidades (0 a 1)
        exp_preds = np.exp(preds - np.max(preds))
        probs = exp_preds / np.sum(exp_preds)
        
        # El modelo FER+ tiene 8 clases: 0:Neutral, 1:Felicidad, 2:Sorpresa, 3:Tristeza, 4:Ira, 5:Disgusto, 6:Miedo, 7:Desprecio
        # Filtramos y mapeamos a nuestras 5 emociones oficiales requeridas
        mapped_probs = {
            "Neutral": probs[0],
            "Felicidad": probs[1],
            "Sorpresa": probs[2],
            "Tristeza": probs[3],
            "Ira": probs[4]
        }
        
        # Normalizamos las probabilidades para que entre las 5 sumen exactamente el 100%
        total_prob = sum(mapped_probs.values())
        confianzas = {}
        for emo, p in mapped_probs.items():
            confianzas[emo] = int((p / total_prob) * 100)
            
        # Determinar la emoción ganadora real
        emocion_predominante = max(confianzas, key=confianzas.get)
        
        # Ajustar para que la suma sea un 100% redondo (evita errores de redondeo como 99% o 101%)
        current_sum = sum(confianzas.values())
        if current_sum != 100:
            confianzas[emocion_predominante] += (100 - current_sum)

        return jsonify({
            "rostros_detectados": len(rostros),
            "emocion_predominante": emocion_predominante,
            "confianzas": confianzas
        }), 200

    except Exception as e:
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)