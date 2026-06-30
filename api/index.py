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

# Cargar clasificadores para heurística de expresiones
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

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
        
        # Lógica Analítica Heurística de Expresiones (Sin IA Externa)
        # Puntuaciones base (predisposición inicial al estado Neutral)
        puntajes = {"Felicidad": 5, "Tristeza": 10, "Ira": 10, "Sorpresa": 5, "Neutral": 40}
        
        # 1. Análisis de Sonrisa (Felicidad)
        # La boca se ubica aproximadamente en el 40% inferior del rostro
        boca_roi = gray_image[y + int(h*0.6):y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(boca_roi, scaleFactor=1.3, minNeighbors=20, minSize=(w//5, h//8))
        
        if len(smiles) > 0:
            puntajes["Felicidad"] += 80
            puntajes["Neutral"] -= 20
            
        # 2. Análisis de Ojos (Sorpresa / Ira)
        # Los ojos se ubican en el tercio medio superior
        ojos_roi = gray_image[y + int(h*0.2):y + int(h*0.55), x:x+w]
        eyes = eye_cascade.detectMultiScale(ojos_roi, scaleFactor=1.1, minNeighbors=7, minSize=(w//6, h//6))
        
        if len(eyes) >= 2:
            # Calcular apertura promedio de los ojos
            alturas_ojos = [e[3] for e in eyes]
            avg_h = sum(alturas_ojos) / len(alturas_ojos)
            proporcion_ojo = avg_h / h
            
            if proporcion_ojo > 0.16: # Ojos muy abiertos (indicador clásico de Sorpresa)
                puntajes["Sorpresa"] += 70
                puntajes["Neutral"] -= 10
            elif proporcion_ojo < 0.11: # Ojos entrecerrados (Ira profunda o tristeza)
                puntajes["Ira"] += 40
                puntajes["Tristeza"] += 20
                
        # 3. Análisis del Ceño (Ira) usando sombras e intensidad de píxeles
        # El entrecejo está entre los ojos
        entrecejo_roi = gray_image[y + int(h*0.15):y + int(h*0.3), x + int(w*0.3):x + int(w*0.7)]
        if entrecejo_roi.size > 0:
            # Convertir a binario resaltando las sombras fuertes del ceño fruncido
            _, thresh = cv2.threshold(entrecejo_roi, 60, 255, cv2.THRESH_BINARY_INV)
            sombra_entrecejo = cv2.countNonZero(thresh) / entrecejo_roi.size
            if sombra_entrecejo > 0.3: # Si hay mucha sombra en el ceño
                puntajes["Ira"] += 50
                puntajes["Neutral"] -= 15
                
        # 4. Asegurar que no existan puntajes negativos y dar fluidez
        mapped_probs = {}
        for emo in puntajes:
            # Añadir pequeña variabilidad orgánica (ruido simulado) para que se vea en vivo
            mapped_probs[emo] = max(1, puntajes[emo] + random.randint(1, 5))
        
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