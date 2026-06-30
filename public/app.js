const EMOJI_MAP = { // Diccionario estático de emparejamiento conceptual de caracteres
    "Felicidad": "😊", "Tristeza": "😢", "Ira": "😡", "Sorpresa": "😮", "Neutral": "😐" // Estructura clave-valor para la representación visual
}; // Cierre del bloque de mapeo gráfico corporativo

const video = document.getElementById('video'); // Enlaza el elemento HTML encargado de reproducir la transmisión multimedia
const cameraPlaceholder = document.getElementById('camera-placeholder'); // Recupera el contenedor gráfico de espera inicial del dashboard
const btnStart = document.getElementById('btn-start'); // Captura el botón disparador del hardware de captura de video
const btnCapture = document.getElementById('btn-capture'); // Obtiene el control encargado de procesar la fotografía fija
const fileInput = document.getElementById('file-input'); // Enlaza el input encargado del procesamiento de archivos locales
const loading = document.getElementById('loading'); // Instancia el indicador visual de carga de peticiones asíncronas
const errorMessage = document.getElementById('error-message'); // Vincula el contenedor encargado de desplegar las notificaciones de fallo
const resultsBox = document.getElementById('results-box'); // Captura el panel principal donde se renderizan los datos métricos
const idleMessage = document.getElementById('idle-message'); // Recupera el elemento descriptivo de estado en espera de acción
const emojiDisplay = document.getElementById('emoji-display'); // Enlaza el nodo de texto destinado a mostrar el emoji representativo
const emotionTitle = document.getElementById('emotion-title'); // Vincula la etiqueta semántica del encabezado de la emoción detectada
const barsContainer = document.getElementById('bars-container'); // Obtiene el nodo contenedor de las barras de progreso dinámicas

let capturedImage = document.getElementById('captured-image'); // Verifica la existencia previa de la etiqueta para congelados estáticos
if (!capturedImage) { // Evaluación condicional para la inyección de elementos en caliente
    capturedImage = document.createElement('img'); // Crea de forma dinámica un nuevo objeto de imagen en el DOM
    capturedImage.id = 'captured-image'; // Configura el atributo ID para asegurar el control exclusivo del elemento
    capturedImage.className = 'w-full h-full object-cover rounded-lg hidden'; // Inyecta las clases CSS de Tailwind para coincidir con la interfaz
    video.parentNode.insertBefore(capturedImage, video.nextSibling); // Posiciona la imagen inmediatamente al lado de la etiqueta de video
} // Cierre del bloque de inicialización dinámica de interfaz

let streamInstance = null; // Puntero global para almacenar el control del flujo multimedia activo

btnStart.addEventListener('click', async () => { // Escucha el evento de click sobre el disparador de activación de cámara
    try { // Bloque de seguridad para procesar respuestas asíncronas de hardware
        errorMessage.classList.add('hidden'); // Limpia y oculta cualquier notificación de alerta previa en la UI
        capturedImage.classList.add('hidden'); // Asegura la ocultación de capturas previas guardadas en el búfer visual
        streamInstance = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } }); // Solicita acceso directo al hardware con dimensiones estándar
        video.srcObject = streamInstance; // Enlaza el flujo binario obtenido de la cámara al reproductor del documento
        video.classList.remove('hidden'); // Transforma el estado del reproductor haciéndolo visible al usuario
        cameraPlaceholder.classList.add('hidden'); // Remueve de la pantalla el fondo gris o logo provisional de espera
        btnCapture.disabled = false; // Remueve el bloqueo del botón disparador habilitando la captura de datos
        btnStart.textContent = "Cámara Activa"; // Modifica la etiqueta de texto del botón para reflejar el estado actual
    } catch (err) { // Captura denegaciones de permisos o fallos de conexión del dispositivo físico
        showError("Error al acceder a la cámara."); // Lanza la alerta en pantalla notificando la restricción de acceso
    } // Cierre de la estructura de manejo de excepciones
}); // Cierre del controlador de eventos del botón de arranque

btnCapture.addEventListener('click', () => { // Registra el evento encargado de procesar la instantánea fotográfica
    if (!streamInstance) return; // Detiene de forma automática el flujo si el hardware de captura no está inicializado

    const canvas = document.createElement('canvas'); // Declara un elemento de renderizado en memoria interna de forma aislada
    canvas.width = video.videoWidth; // Clona las dimensiones horizontales del fotograma nativo del video
    canvas.height = video.videoHeight; // Clona las dimensiones verticales del fotograma nativo del video
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height); // Dibuja una réplica exacta del instante actual en el lienzo
    
    const base64Image = canvas.toDataURL('image/jpeg'); // Serializa los datos del dibujo convirtiéndolos a texto en formato Base64

    capturedImage.src = base64Image; // Asigna la cadena de texto como origen de datos para la visualización estática
    capturedImage.classList.remove('hidden'); // Muestra la foto fija simulando el congelamiento instantáneo de la interfaz
    video.classList.add('hidden'); // Oculta la reproducción en vivo para evitar la persistencia del flujo activo
    
    sendImageToBackend(base64Image); // Despacha los datos codificados hacia el servicio de análisis de Python
}); // Cierre del controlador de eventos de captura manual

fileInput.addEventListener('change', (e) => { // Escucha cambios en el explorador de archivos locales del equipo
    const file = e.target.files[0]; // Captura el primer elemento binario seleccionado por el usuario en la lista
    if (!file) return; // Cancela la operación de procesamiento si el cuadro de diálogo fue cancelado
    const reader = new FileReader(); // Inicializa un objeto lector especializado en flujos de datos locales
    reader.onload = (event) => { // Define el comportamiento una vez la lectura local finalice con éxito
        capturedImage.src = event.target.result; // Carga la imagen procesada localmente directamente al contenedor estático
        capturedImage.classList.remove('hidden'); // Muestra la fotografía del archivo en el recuadro principal
        video.classList.add('hidden'); // Oculta la alimentación en tiempo real de la webcam
        sendImageToBackend(event.target.result); // Envía los datos convertidos hacia la API del backend
    }; // Cierre de la definición de carga de datos locales
    reader.readAsDataURL(file); // Activa la lectura del archivo transformándolo a cadena estandarizada Base64
}); // Cierre del controlador de carga de archivos

async function sendImageToBackend(base64String) { // Función asíncrona dedicada a la comunicación en red con la API Flask
    loading.classList.remove('hidden'); // Activa la visualización de la animación de espera en la interfaz
    errorMessage.classList.add('hidden'); // Oculta avisos de fallos para mantener la consistencia visual
    resultsBox.classList.add('hidden'); // Esconde los paneles de resultados de transacciones previas
    idleMessage.classList.add('hidden'); // Oculta el mensaje informativo por defecto de la aplicación

    try { // Apertura de bloque de seguridad para transacciones de red HTTP
        const response = await fetch('/api/predict', { // Realiza la petición Fetch hacia el endpoint local de Python
            method: 'POST', // Configura el método de comunicación de tipo envío de datos persistentes
            headers: { 'Content-Type': 'application/json' }, // Define el encabezado indicando el envío de datos estructurados en formato JSON
            body: JSON.stringify({ image: base64String }) // Empaqueta el string Base64 dentro del cuerpo de la petición HTTP
        }); // Espera la finalización de la transferencia de datos

        const data = await response.json(); // Parsea el cuerpo de la respuesta HTTP convirtiéndolo en un objeto nativo de JS
        loading.classList.add('hidden'); // Desactiva de inmediato la animación visual del cargador de peticiones

        if (data.error) { // Verifica si el controlador lógico del backend reportó una anomalía funcional
            resetCameraView(); // Restaura la alimentación en vivo de la interfaz web de manera preventiva
            showError(data.error); // Pinta el mensaje exacto enviado por Python en la consola de errores visuales
            return; // Interrumpe el flujo de la función para evitar lecturas sobre datos corruptos o nulos
        } // Cierre del bloque de validación de respuestas de error de API

        renderResults(data); // Invoca al motor de renderizado local pasando las métricas calculadas con éxito
        
        setTimeout(resetCameraView, 3000); // Configura un temporizador asíncrono para restaurar la cámara pasados 3 segundos exactos

    } catch (err) { // Captura fallos críticos de conectividad de red, servidores apagados o puertos bloqueados
        loading.classList.add('hidden'); // Apaga el estado de carga visual en el panel del dashboard
        resetCameraView(); // Asegura el retorno del video en directo para permitir nuevos intentos rápidos
        showError("Error de conexión con el servidor."); // Despliega la advertencia sobre fallos de conectividad con la API externa
    } // Cierre del bloque de control de errores de red
} // Cierre de la definición de la función de despacho en red

function resetCameraView() { // Función de utilidad dedicada a la normalización de la interfaz multimedia
    if (streamInstance) { // Valida que exista una instancia válida y activa de flujo multimedia corriendo
        capturedImage.classList.add('hidden'); // Oculta la fotografía estática del congelado de la pantalla
        video.classList.remove('hidden'); // Vuelve a hacer visible el flujo dinámico de video en vivo de la webcam
    } // Cierre de la condición de existencia de flujo multimedia
} // Cierre de la función de reinicio de la interfaz

function renderResults(data) { // Función modular encargada de la inyección de resultados métricos sobre el árbol DOM
    resultsBox.classList.remove('hidden'); // Remueve el bloqueo de visualización del panel de analíticas estadísticas
    const determinarEmocion = data.emocion_predominante; // Almacena el identificador textual de la emoción ganadora del proceso
    emojiDisplay.textContent = EMOJI_MAP[determinarEmocion] || "😐"; // Actualiza el contenedor inyectando el emoji correspondiente
    emotionTitle.textContent = determinarEmocion; // Escribe el nombre legible de la emoción en el encabezado principal
    barsContainer.innerHTML = ""; // Ejecuta una limpieza total de barras antiguas del contenedor para evitar duplicados

    for (const [emocion, porcentaje] of Object.entries(data.confianzas)) { // Desestructura el diccionario de confianza iterando par por par
        const esPredominante = emocion === determinarEmocion; // Evalúa si la categoría actual coincide con el estado ganador
        const barHtml = ` 
            <div>
                <div class="flex justify-between text-xs mb-1 ${esPredominante ? 'text-teal-400 font-bold' : 'text-slate-400'}">
                    <span>${EMOJI_MAP[emocion]} ${emocion}</span>
                    <span>${porcentaje}%</span>
                </div>
                <div class="w-full bg-slate-700 h-2.5 rounded-full overflow-hidden">
                    <div class="${esPredominante ? 'bg-teal-400' : 'bg-slate-500'} h-2.5 rounded-full" style="width: ${porcentaje}%"></div>
                </div>
            </div>
        `; // Genera la cadena de texto con la estructura HTML y estilos dinámicos de la barra estadística
        barsContainer.insertAdjacentHTML('beforeend', barHtml); // Inserta la barra generada en caliente al final de la lista del panel
    } // Cierre del ciclo de construcción del árbol de elementos
} // Cierre del módulo de inyección estadística de resultados

function showError(msg) { // Módulo estándar encargado del manejo visual de alertas de error
    errorMessage.textContent = msg; // Sobrescribe el texto de la etiqueta con la descripción detallada del problema
    errorMessage.classList.remove('hidden'); // Quita el bloqueo de ocultación haciendo visible la franja de alerta
    idleMessage.classList.remove('hidden'); // Retorna el mensaje neutral de espera de instrucciones en pantalla
    resultsBox.classList.add('hidden'); // Esconde el panel analítico para evitar lecturas erróneas con datos desactualizados
} // Cierre de la función gestora de alertas visuales