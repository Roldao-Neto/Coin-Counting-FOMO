from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import random
from picamera2 import Picamera2  # Importa a biblioteca
import time

app = Flask(__name__)

# --- Variáveis Globais para Contagem ---
# Dicionário para armazenar a contagem de moedas e o total
# Usamos um lock para evitar problemas de concorrência
# (a thread do vídeo escreve, a thread /data lê)
data_lock = threading.Lock()
global_data = {
    "counts": {
        10: 0,
        50: 0,
        100: 0
    },
    "total": 0.0
}

# --- Inicialização da Picamera2 ---
picam2 = Picamera2()
# Configura a câmera para um formato compatível com OpenCV (BGR)
# Ajuste o "size" conforme necessário para performance
config = picam2.create_video_configuration(main={"format": "BGR888", "size": (640, 480)})
picam2.configure(config)
print("Iniciando a PiCamera...")
picam2.start()
time.sleep(1.0) # Pausa para a câmera aquecer e ajustar o auto-foco/exposição
print("PiCamera pronta.")

def processar_frame(frame):
    """
    Função onde você deve colocar sua lógica de ML.
    Esta função recebe um frame do OpenCV (array numpy)
    e deve retornar um dicionário com a contagem de cada moeda.
    """
    
    # ----- INÍCIO DO PLACEHOLDER -----
    # Substitua esta lógica pelo seu modelo de ML
    
    counts = {
        10: random.randint(0, 5),
        50: random.randint(0, 5),
        100: random.randint(0, 5)
    }

    # ----- FIM DO PLACEHOLDER -----
    
    return counts

def calcular_total(counts):
    total = (counts[10] * 0.10) + \
            (counts[50] * 0.50) + \
            (counts[100] * 1.00)
    return total

def gen_frames():
    """Gera frames para o stream de vídeo a partir da Picamera2."""
    global global_data
    while True:
        frame = picam2.capture_array()
        
        if frame is None:
            print("Falha ao capturar frame da PiCamera.")
            continue
            
        # 1. Processa o frame com seu modelo de ML
        counts = processar_frame(frame)
        
        # 2. Calcula o total
        total = calcular_total(counts)
        
        # 3. Atualiza os dados globais de forma segura (thread-safe)
        with data_lock:
            global_data["counts"] = counts
            global_data["total"] = total
        
        # 4. Codifica o frame como JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # 5. Envia o frame para o navegador
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Routes:

@app.route('/')
def index():
    """Rota principal que renderiza o HTML."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Rota que fornece o stream de vídeo."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data():
    """Rota que fornece os dados de contagem em JSON."""
    with data_lock:
        # Retorna uma cópia dos dados globais
        return jsonify(global_data)

if __name__ == '__main__':
    try:
        # 'host=0.0.0.0' torna o servidor visível na sua rede local
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
    finally:
        # Garante que a câmera pare ao encerrar o script
        print("Parando a PiCamera...")
        picam2.stop()
        print("PiCamera parada.")