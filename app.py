from flask import Flask, render_template, Response, jsonify
import threading
import io
import time
import numpy as np
from picamera2 import Picamera2
import cv2

app = Flask(__name__)
model_path = "./ei-contador_moedas_final-lite-float.lite"

# --- Variáveis Globais para Contagem ---
# Dicionário para armazenar a contagem de moedas e o total
# Usamos um lock para evitar problemas de concorrência
# (a thread do vídeo escreve, a thread /data lê)
data_lock = threading.Lock()
global_data = {"counts": {10: 0, 50: 0, 100: 0}, "total": 0.0}

# --- Inicialização da Picamera2 ---
picam2 = None
frame = None
frame_lock = threading.Lock()


def initialize_camera():
    global picam2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Aguarda a câmera aquecer


def get_frame():
    global frame
    while True:
        stream = io.BytesIO()
        picam2.capture_file(stream, format="jpeg")
        with frame_lock:
            frame = stream.getvalue()
        time.sleep(0.1)


def get_latest_frame_array():
    # Converte o último frame JPEG para array numpy para processamento ML
    with frame_lock:
        if frame is None:
            return None
        arr = np.frombuffer(frame, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Corrige para RGB
        return img


# --- Carregar o modelo TFLite ---
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter

interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Parâmetros do modelo FOMO ---
# Ajuste conforme seu modelo (ex: 96x96, 3 canais, etc)
FOMO_INPUT_SHAPE = input_details[0]["shape"]  # [1, h, w, 3]
FOMO_HEIGHT = FOMO_INPUT_SHAPE[1]
FOMO_WIDTH = FOMO_INPUT_SHAPE[2]
FOMO_CHANNELS = FOMO_INPUT_SHAPE[3]
FOMO_CLASSES = output_details[0]["shape"][-1]  # número de classes (inclui background)

# Mapeamento de índices de classe para valores de moeda
# Ajuste conforme sua ordem de classes no treinamento
CLASS_TO_VALUE = {
    1: 10,  # classe 1 = moeda de 10 centavos
    2: 100,  # classe 2 = moeda de 50 centavos
    3: 50,  # classe 3 = moeda de 1 real
    # classe 0 = background
}

# --- Parâmetros para o grid de frequência ---
FREQ_DECAY = 0.9  # Decaimento por frame (0 < FREQ_DECAY < 1)
FREQ_INC = 1.0  # Quanto soma quando detecta uma moeda na célula
FREQ_THRESHOLD = 2.0  # Só mostra moeda se frequência >= esse valor

# Inicializa o grid de frequência: shape [grid_h, grid_w, FOMO_CLASSES]
freq_grid = None


def processar_frame(frame):
    """
    Processa o frame usando o modelo FOMO TFLite e retorna a contagem de moedas
    e uma lista de centroides [(x, y, class_idx), ...].
    Apenas faz a inferência e pega a classe de maior score por célula.
    """
    # 1. Redimensiona e converte para RGB
    img = cv2.resize(frame, (FOMO_WIDTH, FOMO_HEIGHT))
    img = np.expand_dims(img, axis=0)  # [1, h, w, 3]

    # 2. Normaliza conforme tipo do modelo
    input_type = input_details[0]["dtype"]
    if input_type == np.float32:
        img = img.astype(np.float32) / 255.0
    elif input_type == np.int8:
        scale, zero_point = input_details[0].get("quantization", (1.0, 0))
        img = img.astype(np.float32) / 127.0
        img = img / scale + zero_point
        img = np.clip(np.round(img), -128, 127).astype(np.int8)
    else:
        raise ValueError(f"Tipo de input não suportado: {input_type}")

    # 3. Passa pelo modelo
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    output = np.squeeze(output)  # [grid_h, grid_w, classes]

    # 3b. Desquantiza a saída se necessário
    if output_details[0]["dtype"] == np.int8:
        scale, zero_point = output_details[0].get("quantization", (1.0, 0))
        output = (output.astype(np.float32) - zero_point) * scale

    # 4. Para cada célula do grid, pega a classe com maior score
    grid_h, grid_w = output.shape[:2]
    global freq_grid
    if freq_grid is None:
        freq_grid = np.zeros((grid_h, grid_w, FOMO_CLASSES), dtype=np.float32)
    freq_grid *= FREQ_DECAY

    DETECTION_THRESHOLD = 0.5
    for y in range(grid_h):
        for x in range(grid_w):
            cell_scores = output[y, x]
            cell_scores_no_bg = cell_scores[1:]
            if cell_scores_no_bg.size == 0:
                continue
            max_class_rel = np.argmax(cell_scores_no_bg)
            max_class = max_class_rel + 1
            max_score = cell_scores_no_bg[max_class_rel]
            if max_score >= DETECTION_THRESHOLD:
                freq_grid[y, x, max_class] += FREQ_INC

    counts = {10: 0, 50: 0, 100: 0}
    centroides = []

    # --- Agrupamento de células adjacentes por classe ---
    for class_idx, value in CLASS_TO_VALUE.items():
        # Cria máscara binária para a classe
        mask = (freq_grid[:, :, class_idx] >= FREQ_THRESHOLD).astype(np.uint8)
        if np.count_nonzero(mask) == 0:
            continue
        # Encontra componentes conectados (clusters)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        for i in range(1, num_labels):  # Ignora label 0 (background)
            cx, cy = centroids[i]
            # Calcula bounding box para desenhar elipse/círculo
            x, y, w, h, area = stats[i]
            # Converte centroide para coordenadas da imagem original
            cx_img = int((cx + 0.5) * frame.shape[1] / grid_w)
            cy_img = int((cy + 0.5) * frame.shape[0] / grid_h)
            rx = int(w * frame.shape[1] / grid_w / 2)
            ry = int(h * frame.shape[0] / grid_h / 2)
            centroides.append((cx_img, cy_img, class_idx, rx, ry))
            counts[value] += 1

    return counts, centroides


def calcular_total(counts):
    total = (counts[10] * 0.10) + (counts[50] * 0.50) + (counts[100] * 1.00)
    return total


def gen_frames():
    """Gera frames para o stream de vídeo a partir da Picamera2."""
    global global_data
    while True:
        img = get_latest_frame_array()
        if img is None:
            continue

        # 1. Processa o frame com seu modelo de ML
        counts, centroides = processar_frame(img)

        # 2. Calcula o total
        total = calcular_total(counts)

        # 3. Atualiza os dados globais de forma segura (thread-safe)
        with data_lock:
            global_data["counts"] = counts
            global_data["total"] = total

        # --- DESENHA ELIPSES/CÍRCULOS AGRUPADOS COM ALPHA ---
        overlay = img.copy()
        for cx, cy, class_idx, rx, ry in centroides:
            # Cores RGB distintas para cada classe
            if class_idx == 1:
                color = (255, 255, 0)  # Amarelo para 10 centavos
            elif class_idx == 2:
                color = (0, 0, 255)  # Azul para 50 centavos
            elif class_idx == 3:
                color = (255, 0, 0)  # Vermelho para 1 real
            else:
                color = (0, 255, 0)  # Verde padrão

            ellipse_rx = max(rx, 20)
            ellipse_ry = max(ry, 20)
            if ellipse_rx > 0 and ellipse_ry > 0:
                cv2.ellipse(
                    overlay, (cx, cy), (ellipse_rx, ellipse_ry), 0, 0, 360, color, 3
                )
            else:
                cv2.circle(overlay, (cx, cy), 24, color, 3)

        # Blend overlay com alpha 0.5
        img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

        # Converta para BGR antes de enviar para o navegador
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode(".jpg", frame_bgr)
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


# Routes:


@app.route("/")
def index():
    """Rota principal que renderiza o HTML."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Rota que fornece o stream de vídeo."""
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/data")
def data():
    """Rota que fornece os dados de contagem em JSON."""
    with data_lock:
        # Retorna uma cópia dos dados globais
        return jsonify(global_data)


if __name__ == "__main__":
    initialize_camera()
    threading.Thread(target=get_frame, daemon=True).start()
    try:
        app.run(
            host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False
        )
    finally:
        if picam2:
            picam2.stop()
