# Contagem de Moedas com Modelo FOMO (Coin Counting with FOMO):

Projeto de visão computacional embarcada que utiliza um modelo FOMO (Faster Objects, More Objects) em uma Raspberry Pi Zero 2W para detectar, contar e somar moedas (R$ 0,10, R$ 0,50 e R$ 1,00) em tempo real.

A aplicação serve uma interface web local que exibe o feed de vídeo ao vivo da câmera e atualiza a contagem e o valor total dinamicamente.

## Made by:

- <a href="https://github.com/Roldao-Neto" target="_blank">Roldão Neto</a>
- <a href="https://github.com/iRyanRib" target="_blank">RyanRib</a>
- <a href="https://github.com/rodrigowillsilva" target="_blank">Rodrigo Silva</a>

## Funcionalidades::

- Detecção em Tempo Real: Identifica moedas (10, 50 centavos e 1 Real) usando um modelo FOMO (.lite) via TFLite Runtime.
- Contagem e Soma: Calcula a quantidade de cada moeda e o valor monetário total.
- Interface Web: Exibe o feed de vídeo ao vivo (streaming MJPEG) e os resultados atualizados dinamicamente via Flask.
- Estabilização de Detecção: Utiliza um "grid de frequência" (freq_grid) para estabilizar as detecções e evitar contagens flutuantes (flickering).
- Agrupamento de Detecções: Usa cv2.connectedComponentsWithStats para agrupar células de detecção adjacentes, contando moedas parcialmente cobertas ou detectadas em múltiplos quadrantes como um único objeto.

## Tecnologias Utilizadas:

- **Hardware**: Raspberry Pi Zero 2W, Câmera (Picamera2)

- **Backend**: Python, Flask, TensorFlow Lite Runtime, OpenCV (opencv-python-headless), NumPy

- **Frontend**: HTML, CSS, JavaScript (Vanilla)

- **Modelo de ML**: FOMO (treinado na Edge Impulse)

## Como Funciona:

A aplicação é dividida em um backend (Flask) e um frontend (HTML/CSS/JS).

**Backend** (app.py)
- Servidor Flask: Inicia um servidor web.

- Câmera: A Picamera2 é inicializada. Uma thread (get_frame) captura continuamente imagens da câmera e as armazena na memória (frame).

- Stream de Vídeo (/video_feed):

1) Uma rota de streaming MJPEG (gen_frames) busca o frame mais recente.

2) O frame é processado pela função processar_frame, que executa a inferência do modelo TFLite.

3) A saída do FOMO (um heatmap) é estabilizada pelo freq_grid.

4) cv2.connectedComponentsWithStats agrupa os centroides detectados.

5) Elipses são desenhadas sobre as moedas detectadas e o frame é enviado para o navegador.

- API de Dados (/data):

- Os resultados da contagem (counts) e o total (total) são armazenados em uma variável global (global_data) protegida por um threading.Lock.

- Esta rota simplesmente retorna o conteúdo de global_data em formato JSON.

**Frontend** (index.html, main.js, styles.css)
- HTML: Estrutura a página com um elemento <img> (que aponta para /video_feed) e <span>s para exibir os valores.

- JavaScript: O main.js executa uma função atualizarValores a cada 500ms.

- Fetch API: Essa função faz uma requisição fetch para a rota /data, recebe o JSON e atualiza o conteúdo dos <span>s com os novos valores de contagem e o total formatado (R$).

## Instalação e Execução:

```bash
git clone https://github.com/Roldao-Neto/Coin-Counting-FOMO/
```

```bash
# Na Raspberry Pi:
pip install flask
pip install opencv-python-headless==
pip install numpy==1.24.2
pip install picamera2
pip install tflite-runtime
```

Adicione o seu modelo (ou o nosso) no projeto e atualize o model_path em app.py.

Rode o servidor flask com o seguinte comando no terminal:

```bash
python3 app.py
```
