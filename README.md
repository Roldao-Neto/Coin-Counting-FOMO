# Coin Counting with FOMO:

Embedded Real-time Computer Vision Project with FOMO (Faster Objects, More Objects) in a Raspberry Pi Zero 2W to detect, count, and sum Brazilian coins - R$ 0,10, R$ 0,50 e R$ 1,00.

This Application uses a local web interface to display a live video feed from the camera and dynamically update the positions of the coins and the total value.

## Made by:

- <a href="https://github.com/Roldao-Neto" target="_blank">Roldão Neto</a>
- <a href="https://github.com/iRyanRib" target="_blank">RyanRib</a>
- <a href="https://github.com/rodrigowillsilva" target="_blank">Rodrigo Silva</a>

## Functionalities:

- Real Time Detection: Identify coins using a FOMO model with TFLite Runtime.
- Counting and Total Value: Calculate how many coins we have for each class and update the total value accordingly.
- Web Interface: Display the live video feed (streaming MJPEG) and update the results via Flask.
- Frequency Grid: Stabilize the detection in order to avoid flickering.
- Clustering Detections: Use cv2.connectedComponentsWithStats to group together adjacent detection cells, counting partially covered coins and multiple detections for the same coin as a unique object.

## Technologies:

- **Hardware**: Raspberry Pi Zero 2W with Camera (Picamera2)

- **Backend**: Python, Flask, TensorFlow Lite Runtime, OpenCV (opencv-python-headless), NumPy

- **Frontend**: HTML, CSS, JavaScript (Vanilla)

- **ML Model**: FOMO (Trained with Edge Impulse)

## How it Works:

The App is divided between a backend (Flask) and a frontend (HTML/CSS/JS).

**Backend** (app.py)

- Flask Server: Starts the web server.

- Camera: The Picamera2 is started. A thread (get_frame) captures images from the camera and stores them in memory (frame).

- Video Stream (/video_feed):

1) The streaming route (gen_frames) searches for the most recent frame.

2) The frame is processed by a function called "processar_frame", which makes the inference of the tflite model.

3) The FOMO output (heatmap) is stabilized by freq_grid.

4) cv2.connectedComponentsWithStats groups the detected centroids.

5) Circumferences are drawn over the detected coins, and the frame is sent to the browser.

- Data API (/data)

- The results (counts & total) are stored in a global variable (global_data) protected by a threading.Lock.

**Frontend** (index.html, main.js, styles.css)

- HTML: Structures the page.

- JavaScript: Main.js runs the function "atualizarValores" (update Values) every 500ms.

- Fetch API: This function fetches JSON from /data, receives the JSON, and updates the <span> content with the new counts and the total formatted (R$).

## Installation:

```bash
git clone https://github.com/Roldao-Neto/Coin-Counting-FOMO/
cd Coin-Counting-FOMO
```

```bash
# In the Raspberry Pi (or another device):
pip install -r requirements-rasp.txt
```

Add your own model (or our model) in the project and update the model_path in app.py.

Run the Flask server:

```bash
python3 app.py
```

Access the dashboard at http://[ip]:5000
