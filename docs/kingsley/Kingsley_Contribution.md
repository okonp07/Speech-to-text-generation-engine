# Kingsley's Contribution: Real-time Transcription Engine

## Feature Objective
To implement a high-performance, low-latency real-time transcription engine that utilizes WebSockets for streaming audio and incorporates an AI-driven noise-cleaning layer (DeepFilterNet) before performing inference with `faster-whisper`.

---

## 1. Infrastructure Requirements
To support Approach 2 (Production WebSocket), the following infrastructure and dependencies are required:

*   **FastAPI:** To manage high-concurrency WebSocket connections and handle asynchronous audio streams.
*   **WebSockets (websockets library):** To enable full-duplex communication between the client (browser) and the backend.
*   **DeepFilterNet / RNNoise:** To provide a robust, AI-powered noise suppression layer that strips background noise from raw audio chunks.
*   **Faster-Whisper (Async Integration):** To perform transcription on sliding audio buffers without blocking the main event loop.
*   **Redis (Optional):** To store session state and manage long-running transcription tasks if scaling beyond a single instance.
*   **Uvicorn:** A lightning-fast ASGI server to host the FastAPI application.

---

## 2. Feature Implementation Path (Weekly Tasks)

### Week 1: WebSocket Core & STT Boilerplate
*   **Task 1.1:** Setup a FastAPI server with a basic `/ws/transcribe` endpoint.
*   **Task 1.2:** Implement an asynchronous audio buffer that can receive binary chunks (PCM16/Float32) from a client.
*   **Task 1.3:** Integrate a basic `faster-whisper` worker that transcribes the current buffer whenever a "silent" segment is detected or a time threshold is met.
*   **Task 1.4:** Build a simple test client (HTML/JS) to verify binary data transmission over WebSockets.

### Week 2: AI Noise Suppression Layer
*   **Task 2.1:** Integrate `DeepFilterNet` into the audio processing pipeline in `digit_recognition/audio.py`.
*   **Task 2.2:** Implement real-time chunk-by-chunk cleaning of the incoming WebSocket audio stream.
*   **Task 2.3:** Benchmark the latency impact of the noise reduction layer and optimize the model if necessary (e.g., using a smaller DF-Net variant).

### Week 3: Frontend Integration & VAD
*   **Task 3.1:** Integrate `streamlit-webrtc` or a custom React/JS component into `streamlit_app.py` to capture and stream microphone audio.
*   **Task 3.2:** Implement Voice Activity Detection (VAD) on the server-side to prevent transcription when no speech is detected.
*   **Task 3.3:** Design the UI for "partial" vs "final" transcripts (displaying unstable intermediate text as the user speaks).

### Week 4: Optimization, Stability & Final Testing
*   **Task 4.1:** Implement "Sliding Window" context (passing previous transcript context to Whisper to improve accuracy of the current chunk).
*   **Task 4.2:** Optimize CPU/Memory usage to handle multiple concurrent WebSocket sessions.
*   **Task 4.3:** Perform end-to-end stress testing and finalize the `weekly-log` documentation.

---

## 3. Weekly Log Directory
All progress reports, bug logs, and performance metrics for the above tasks should be stored in:
`docs/kingsley/weekly-log/week_X_log.md`
