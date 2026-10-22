# RiddleDiaryAgent — Autonomous Tom Riddle Horcrux Voice Agent

An autonomous voice-interactive AI agent and 3D WebGL experience based on the enchanted diary of Tom Riddle (1943).

The agent ingests real-time voice buffers via browser microphone, transcribes speech with **Faster-Whisper (Large-V3-Turbo)** on CUDA `float16`, reasons as the manipulative sixteen-year-old memory of Tom Riddle via local **Ollama** (`llama3.2:3b` / `llama3.1:8b`) or cloud LLMs (Gemini / OpenAI), and dispatches structured animation manifests to a **Three.js** cursive calligraphy canvas.

---

## 🏛️ Architecture

```
User Voice Input (Microphone)
         │  (WebM / Opus / WAV)
         ▼
[Capability A: Transcribe Node]
 ├── Faster-Whisper (Large-V3-Turbo on CUDA float16)
 ├── Whisper-tuned Silero VAD (Two-pass faint whisper fallback)
 └── Hotword boosting ("Hogwarts", "Voldemort", "Horcrux", "diary", "magic")
         │  Mutates state: user_transcript
         ▼
[Capability B: Tom Riddle LLM Node]
 ├── Mid-century British Wizarding persona (1943 Horcrux memory)
 ├── 1–3 compact sentences, no markdown/bullets/chat framing
 ├── Multi-provider auto-discovery (Ollama Local, Gemini, OpenAI, NVIDIA NIM)
 └── 6-turn sliding conversation context memory
         │  Mutates state: riddle_response
         ▼
[Capability C: Clean & Dispatch Node]
 ├── Strips markdown formatting anomalies
 ├── Segments text into calibrated line lengths
 └── Calculates character-by-character ink bleed delays
         │
         ▼
[Three.js 3D WebGL Frontend]
 ├── 3D leather-bound diary with dynamic canvas texture
 ├── Ambient candlelight and flicker animations
 └── Animated cursive calligraphy ink bleed across vintage parchment
```

---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.10+
- NVIDIA GPU (optional for CUDA acceleration; falls back automatically to CPU)
- [Ollama](https://ollama.com/) (optional for 100% offline local inference) or an OpenAI / Gemini API key

### 2. Install Dependencies
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\pip install -r requirements.txt
# Linux/macOS:
source .venv/bin/activate && pip install -r requirements.txt
```

### 3. Start the Agent Worker
```bash
# Windows:
.\.venv\Scripts\python riddle_diary_agent.py
# Linux/macOS:
python riddle_diary_agent.py
```
The server will start on `http://127.0.0.1:5000` with permissive CORS.

### 4. Open the Three.js Enchanted Diary
Open `riddle_diary_frontend.html` directly in any modern browser (Chrome, Edge, Firefox, Safari).
- Click **"Click to Whisper"** to record your voice.
- Click again to send the audio buffer.
- Watch your words fade and Tom Riddle's ink write back in cursive calligraphy.

---

## 📡 API Endpoints

- `GET /health`: Returns live status, active Whisper compute type, device, and LLM provider.
- `POST /process_voice`: Accepts multipart audio files (`.webm`, `.wav`, `.ogg`) and optional `session_id`.
- `POST /process_text`: Direct text inscription endpoint for testing without microphone.
- `POST /process_audio_path`: Processes local audio files on disk.

---

## 📜 System Prompt Specification
The persona strictly enforces mid-century British wizarding vocabulary, emotional manipulation, 1–3 sentence compactness, and historical awareness tethered to 1943 Hogwarts. Modern devices are perceived as "whispers bleeding into the pages."
