import uvicorn
import asyncio
import numpy as np
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from digit_recognition import SpeechTranscriber

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Task 1.1 & 1.3: Initialize Transcriber
# We use 'base' for a good balance of speed and accuracy
try:
    transcriber = SpeechTranscriber(model_size="base")
    logger.info("SpeechTranscriber initialized with 'base' model")
except Exception as e:
    logger.error(f"Failed to initialize SpeechTranscriber: {e}")
    transcriber = None

# Task 1.4: Simple HTML Client for testing
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>WebSocket Transcription Test</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
            #transcript { border: 1px solid #ddd; padding: 1rem; min-height: 200px; margin-top: 1rem; white-space: pre-wrap; background: #f9f9f9; }
            .controls { display: flex; gap: 1rem; align-items: center; }
            .status { font-size: 0.9rem; color: #666; }
        </style>
    </head>
    <body>
        <h1>WebSocket Transcription Test</h1>
        <div class="controls">
            <button id="start">Start Recording</button>
            <button id="stop" disabled>Stop Recording</button>
            <label><input type="checkbox" id="denoise" checked> Denoise Audio</label>
            <span id="status" class="status">Ready</span>
        </div>
        <div id="transcript"></div>
        <script>
            let socket;
            let audioContext;
            let processor;
            let input;

            const startBtn = document.getElementById('start');
            const stopBtn = document.getElementById('stop');
            const denoiseCheckbox = document.getElementById('denoise');
            const transcriptDiv = document.getElementById('transcript');
            const statusSpan = document.getElementById('status');

            denoiseCheckbox.onchange = () => {
                if (socket && socket.readyState === WebSocket.OPEN) {
                    socket.send(JSON.stringify({ denoise: denoiseCheckbox.checked }));
                }
            };

            startBtn.onclick = async () => {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                socket = new WebSocket(`${protocol}//${window.location.host}/ws/transcribe`);
                
                socket.onopen = () => {
                    statusSpan.innerText = "Connected";
                    statusSpan.style.color = "green";
                    // Send initial config
                    socket.send(JSON.stringify({ denoise: denoiseCheckbox.checked }));
                };

                socket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.type === "partial" || data.type === "final") {
                        const tag = data.denoised ? " [Denoised]" : "";
                        transcriptDiv.innerText += data.text + tag + " ";
                    } else if (data.type === "error") {
                        console.error("Server Error:", data.message);
                    }
                };

                socket.onclose = () => {
                    statusSpan.innerText = "Disconnected";
                    statusSpan.style.color = "red";
                };

                try {
                    audioContext = new (window.AudioContext || window.webkitAudioContext)({sampleRate: 16000});
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    input = audioContext.createMediaStreamSource(stream);
                    // ScriptProcessor is deprecated but compatible for simple tests
                    processor = audioContext.createScriptProcessor(4096, 1, 1);

                    processor.onaudioprocess = (e) => {
                        const channelData = e.inputBuffer.getChannelData(0);
                        if (socket.readyState === WebSocket.OPEN) {
                            socket.send(channelData.buffer);
                        }
                    };

                    input.connect(processor);
                    processor.connect(audioContext.destination);

                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                } catch (err) {
                    alert("Could not access microphone: " + err);
                }
            };

            stopBtn.onclick = () => {
                if (processor) {
                    processor.disconnect();
                    input.disconnect();
                }
                if (socket) {
                    socket.close();
                }
                startBtn.disabled = false;
                stopBtn.disabled = true;
                statusSpan.innerText = "Stopped";
            };
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": transcriber is not None}

# Task 1.1 & 1.2: WebSocket Endpoint and Asynchronous Buffer
@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    # Internal buffer to hold audio chunks
    audio_buffer = []
    
    # We'll use a lock to prevent concurrent transcription calls on the same session
    transcribe_lock = asyncio.Lock()
    
    async def transcribe_chunk(audio_np, denoise=True):
        if not transcriber:
            await websocket.send_json({"type": "error", "message": "Transcriber not initialized"})
            return
            
        async with transcribe_lock:
            try:
                # Task 2.2: Apply noise reduction
                processed_audio = audio_np
                if denoise:
                    processed_audio = await asyncio.to_thread(
                        transcriber.processor.denoise, 
                        audio_np, 
                        sample_rate=16000
                    )

                # Transcribe
                result = await asyncio.to_thread(
                    transcriber.transcribe_array, 
                    processed_audio, 
                    sample_rate=16000
                )
                if result.text.strip() and result.text != "No speech detected.":
                    await websocket.send_json({
                        "type": "final", 
                        "text": result.text,
                        "confidence": result.confidence,
                        "denoised": denoise
                    })
            except Exception as e:
                logger.error(f"Transcription/Denoise error: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})

    use_denoise = True

    try:
        while True:
            # Receive data (could be bytes or text for configuration)
            message = await websocket.receive()
            
            if "bytes" in message:
                data = message["bytes"]
                chunk = np.frombuffer(data, dtype=np.float32)
                audio_buffer.append(chunk)
                
                if len(audio_buffer) >= 12:
                    full_audio = np.concatenate(audio_buffer)
                    asyncio.create_task(transcribe_chunk(full_audio, denoise=use_denoise))
                    audio_buffer = []
            elif "text" in message:
                import json
                config = json.loads(message["text"])
                if "denoise" in config:
                    use_denoise = config["denoise"]
                    logger.info(f"Denoising set to: {use_denoise}")
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
