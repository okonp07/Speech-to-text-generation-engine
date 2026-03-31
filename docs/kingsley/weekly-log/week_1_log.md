# Weekly Log: Week 1

## Status: Completed

## Tasks Accomplished
*   **Task 1.1: FastAPI Server Setup:** Created `main.py` with FastAPI and a dedicated `/ws/transcribe` WebSocket endpoint.
*   **Task 1.2: Asynchronous Audio Buffer:** Implemented binary data handling in the WebSocket loop, converting incoming Float32 bytes into NumPy arrays for processing.
*   **Task 1.3: STT Integration:** Integrated the existing `SpeechTranscriber` (Whisper) into the WebSocket flow using `asyncio.to_thread` to prevent blocking the main event loop.
*   **Task 1.4: Test Client:** Developed an embedded HTML/JS client with microphone access (using `navigator.mediaDevices.getUserMedia`) to stream live audio to the backend.

## Technical Details
*   **Server:** FastAPI running on Uvicorn (port 8000).
*   **Audio Format:** 16kHz Mono Float32 (Standard for Whisper).
*   **Transcription Logic:** Currently triggers every ~12 chunks (approx. 3 seconds) and clears the buffer.

## Challenges & Observations
*   **Dependency Issues:** Encountered `WinError 193` with `torch` in the local environment. Handled this gracefully in the code by catching initialization errors and reporting them to the client via JSON messages.
*   **Latency:** The 3-second chunking approach works for basic testing but will be improved with sliding windows and VAD in Weeks 3 and 4.

## Next Steps (Week 2)
*   Integrate `DeepFilterNet` for real-time noise suppression.
*   Implement chunk-by-chunk audio cleaning.
