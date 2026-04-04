# Weekly Log: Week 2

## Status: Completed

## Tasks Accomplished
*   **Task 2.1: DeepFilterNet Integration:** Integrated AI-powered noise suppression into `digit_recognition/audio.py` using `DeepFilterNet`.
*   **Task 2.2: Real-time Denoising:** Updated the FastAPI WebSocket loop to allow chunk-by-chunk audio cleaning before transcription. Added a toggle in the test client to enable/disable denoising.
*   **Task 2.3: Latency Check:** Verified that denoising runs in a background thread (`asyncio.to_thread`) to maintain WebSocket responsiveness. Re-installed `torch` to resolve environment DLL issues.

## Technical Details
*   **Noise Suppression:** DeepFilterNet (DF-Net) is now a core part of the `AudioProcessor`.
*   **Thread Safety:** Transcription and Denoising are now protected by an `asyncio.Lock` per session to prevent race conditions during heavy overlapping chunks.
*   **Client Control:** The WebSocket protocol now supports a JSON-based configuration message to toggle features like `denoise` on-the-fly.

## Challenges & Observations
*   **Model Size:** DeepFilterNet downloads its pre-trained model on the first run, similar to Whisper.
*   **Audio Quality:** Denoising significantly improves Whisper's accuracy in noisy environments (e.g., background fan noise).

## Next Steps (Week 3)
*   Integrate the WebSocket client into the main `streamlit_app.py`.
*   Implement Voice Activity Detection (VAD) on the server side.
