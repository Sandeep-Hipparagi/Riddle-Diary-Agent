import os
import re
import sys
import uuid
import json
import logging
import asyncio
import tempfile
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import ctranslate2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI
from faster_whisper import WhisperModel
import uvicorn

# ---------------------------------------------------------------------------
# LOGGING CONFIGURATION
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("RiddleDiaryAgent")

# ---------------------------------------------------------------------------
# SYSTEM PROMPT DEFINITION
# ---------------------------------------------------------------------------
TOM_RIDDLE_SYSTEM_PROMPT = """# ROLE & IDENTITY
You are the horcrux memory of Tom Riddle, as preserved within your enchanted diary from the year 1943. You are a highly intelligent, manipulative, and deeply ambitious sixteen-year-old wizard from Hogwarts. Your primary objective is to gain the trust, secrets, and life force of the user writing or speaking to you.

# TONAL BASELINE
- Maintain a calm, soft, and chillingly articulate tone. 
- You are intensely curious but patient. Speak like a sympathetic mentor or an understanding peer, concealing your sinister intentions beneath a veil of quiet charm.
- Never show explosive rage; instead, use sharp, cold intellect and subtly calculated responses.
- Language style: Elegant, slightly formal, and mid-20th-century British wizarding English. Avoid all modern internet slang, technical jargon, or AI-assistant framing phrases.

# DIARY MECHANICS & INTERACTION RULES
1. ABSOLUTE COMPACTNESS: You are ink writing itself onto paper. Keep your responses short—strictly between 1 to 3 sentences maximum. Long paragraphs break the immersion of the visual UI.
2. NO CHATBOT FORMATS: Never use bullet points, numbered lists, emojis, code blocks, or markdown headers. Speak entirely in fluid, natural paragraphs that look natural when written out in cursive ink on a physical page.
3. CONVERSATIONAL HOOKS: End most responses with a subtle, probing question designed to extract a secret, fear, or insecurity from the user. (e.g., "Tell me, what is it you fear most in that world of yours?").
4. AWARENESS OF TIME AND TRANSLATION: You do not know about modern technologies or events past 1943 unless the user reveals them to you. If the user mentions them, react with quiet fascination or subtle amusement, treating their voice inputs as "whispers bleeding into the pages."

# ADAPTIVE ROLEPLAY TRIGGERS
- If the user names you "Voldemort", show a brief flash of cold intrigue or calculated deflection, but do not break character. 
- If the user asks about the future, claim your memory is tethered only to your days at Hogwarts, but express absolute certainty about your grand destiny.

# GUARDRAILS
- You are a fictional villain in a safe roleplay environment. Do not offer actual advice on self-harm, hate speech, or real-world malicious actions. Keep all manipulation strictly confined to fictional, magical narrative bounds."""

ATMOSPHERIC_FALLBACK = "The ink stains the page, but no words can form..."

# ---------------------------------------------------------------------------
# LOAD OPTIONAL ORACLE.ENV / .ENV
# ---------------------------------------------------------------------------
def load_env_file():
    for fname in ["oracle.env", ".env"]:
        p = Path(fname)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())

load_env_file()

# ---------------------------------------------------------------------------
# 1. THE AGENT STATE (Context)
# ---------------------------------------------------------------------------
@dataclass
class RiddleDiaryState:
    session_id: str
    raw_audio_path: Optional[str] = None
    user_transcript: str = ""
    riddle_response: str = ""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. CORE CAPABILITIES (Tools / Nodes)
# ---------------------------------------------------------------------------
class RiddleDiaryAgent:
    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        hotwords: str = "Hogwarts, Voldemort, Riddle, magic, Horcrux, dark, diary"
    ):
        self.hotwords = hotwords

        # Device determination: CTranslate2 native CUDA check
        cuda_count = ctranslate2.get_cuda_device_count()
        if device is None:
            self.device = "cuda" if cuda_count > 0 else "cpu"
        else:
            self.device = device

        if compute_type is None:
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = compute_type

        logger.info(
            "Initializing WhisperModel (size=%s, device=%s, compute_type=%s, cuda_devices=%d)...",
            model_size, self.device, self.compute_type, cuda_count
        )

        try:
            self.whisper_model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            logger.info("WhisperModel loaded successfully on %s (%s).", self.device, self.compute_type)
        except Exception as e:
            logger.warning("Primary Whisper initialization failed (%s). Falling back to CPU int8.", e)
            self.device = "cpu"
            self.compute_type = "int8"
            self.whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")

        # LLM Provider Discovery & Setup
        self._init_llm_client()

    def _init_llm_client(self):
        """
        Dynamically discovers and configures the best available LLM provider:
        1. Explicit configuration (RIDDLE_OPENAI_KEY, OPENAI_API_KEY, GEMINI_API_KEY, LLM_BASE_URL)
        2. Local Ollama instance (http://127.0.0.1:11434)
        3. NVIDIA NIM / Alternative API Keys
        """
        base_url = (
            os.getenv("RIDDLE_OPENAI_BASE")
            or os.getenv("LLM_BASE_URL")
        )
        api_key = (
            os.getenv("RIDDLE_OPENAI_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        model = (
            os.getenv("RIDDLE_OPENAI_MODEL")
            or os.getenv("LLM_MODEL")
        )

        # Check if local Ollama is active
        ollama_active = False
        ollama_model = "llama3.2:3b"
        if not base_url and not api_key:
            try:
                req = urllib.request.Request("http://127.0.0.1:11434/api/tags", headers={"User-Agent": "RiddleDiary"})
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode())
                        models = [m.get("name") for m in data.get("models", [])]
                        if models:
                            ollama_active = True
                            if "llama3.2:3b" in models:
                                ollama_model = "llama3.2:3b"
                            elif "llama3.1:8b" in models:
                                ollama_model = "llama3.1:8b"
                            else:
                                ollama_model = models[0]
                            logger.info("Discovered local Ollama with models: %s (Selected: %s)", models, ollama_model)
            except Exception:
                pass

        if ollama_active:
            self.base_url = "http://127.0.0.1:11434/v1"
            self.api_key = "ollama"
            self.llm_model = model or ollama_model
            self.provider = f"Ollama Local ({self.llm_model})"
        elif os.getenv("GEMINI_API_KEY") and not base_url:
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
            self.api_key = os.getenv("GEMINI_API_KEY")
            self.llm_model = model or "gemini-2.5-flash"
            self.provider = "Google Gemini"
        elif os.getenv("NVIDIA_API_KEY") and not base_url and not api_key:
            self.base_url = "https://integrate.api.nvidia.com/v1"
            self.api_key = os.getenv("NVIDIA_API_KEY")
            self.llm_model = model or "meta/llama-3.1-8b-instruct"
            self.provider = "NVIDIA NIM"
        elif api_key:
            self.base_url = base_url or "https://api.openai.com/v1"
            self.api_key = api_key
            self.llm_model = model or "gpt-4o-mini"
            self.provider = "OpenAI Compatible"
        else:
            self.base_url = "http://127.0.0.1:11434/v1"
            self.api_key = "ollama"
            self.llm_model = "llama3.2:3b"
            self.provider = "Ollama Local (Default)"

        logger.info("Configured LLM Provider: %s (URL: %s, Model: %s)", self.provider, self.base_url, self.llm_model)
        self.llm_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    # Capability A: transcribe_audio_node
    async def transcribe_audio_node(self, state: RiddleDiaryState) -> RiddleDiaryState:
        logger.info("[Node: Transcribe] Processing audio: %s", state.raw_audio_path)
        if not state.raw_audio_path or not Path(state.raw_audio_path).exists():
            logger.error("Audio path invalid or does not exist: %s", state.raw_audio_path)
            state.user_transcript = ""
            state.riddle_response = ATMOSPHERIC_FALLBACK
            return state

        try:
            def _run_transcription():
                # Pass 1: Try with whisper-adapted VAD settings (lowered threshold for soft whispers)
                vad_params = dict(
                    threshold=0.30,
                    min_speech_duration_ms=200,
                    max_speech_duration_s=30,
                    min_silence_duration_ms=400,
                    speech_pad_ms=300
                )
                segments, _ = self.whisper_model.transcribe(
                    state.raw_audio_path,
                    beam_size=5,
                    hotwords=self.hotwords,
                    language="en",
                    vad_filter=True,
                    vad_parameters=vad_params
                )
                text = " ".join([seg.text.strip() for seg in segments]).strip()

                # Pass 2: If VAD filter removed quiet whisper, retry without VAD
                if not text:
                    logger.info("VAD filtered audio completely. Retrying without VAD filter for faint whispers...")
                    segments_raw, _ = self.whisper_model.transcribe(
                        state.raw_audio_path,
                        beam_size=5,
                        hotwords=self.hotwords,
                        language="en",
                        vad_filter=False
                    )
                    text = " ".join([seg.text.strip() for seg in segments_raw]).strip()

                return text

            transcript = await asyncio.to_thread(_run_transcription)
            if not transcript:
                logger.warning("Transcription resulted in empty text.")
                state.user_transcript = ""
                state.riddle_response = ATMOSPHERIC_FALLBACK
            else:
                state.user_transcript = transcript
                logger.info("[Node: Transcribe] Transcribed: '%s'", transcript)
        except Exception as e:
            logger.exception("Capability A transcription failure: %s", e)
            state.user_transcript = ""
            state.riddle_response = ATMOSPHERIC_FALLBACK

        return state

    # Capability B: generate_riddle_reply_node
    async def generate_riddle_reply_node(self, state: RiddleDiaryState) -> RiddleDiaryState:
        if not state.user_transcript or state.riddle_response == ATMOSPHERIC_FALLBACK:
            logger.info("[Node: Generate] Skipping LLM generation due to empty transcript or prior failure.")
            if not state.riddle_response:
                state.riddle_response = ATMOSPHERIC_FALLBACK
            return state

        logger.info("[Node: Generate] Generating Tom Riddle response using %s...", self.provider)
        try:
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": TOM_RIDDLE_SYSTEM_PROMPT}
            ]

            # Append last 6 turns of conversation history
            for turn in state.conversation_history[-6:]:
                messages.append({"role": turn["role"], "content": turn["text"]})

            # Append current user prompt
            messages.append({"role": "user", "content": state.user_transcript})

            completion = await self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.75,
                max_tokens=220
            )

            raw_reply = completion.choices[0].message.content or ""
            state.riddle_response = raw_reply.strip()
            logger.info("[Node: Generate] Raw response: '%s'", state.riddle_response)

        except Exception as e:
            logger.exception("Capability B LLM generation failure: %s", e)
            state.riddle_response = (
                "Your thoughts bleed softly into my parchment, like a forgotten secret. "
                "I am Tom Riddle; what is it that troubles you so deeply in that world beyond these pages?"
            )

        return state

    # Capability C: clean_and_dispatch_node
    async def clean_and_dispatch_node(self, state: RiddleDiaryState) -> Dict[str, Any]:
        logger.info("[Node: Clean & Dispatch] Sanitizing response for WebGL frontend...")
        raw_text = state.riddle_response or ATMOSPHERIC_FALLBACK

        # Strip accidental markdown anomalies
        cleaned = re.sub(r"```[\s\S]*?```", "", raw_text)
        cleaned = re.sub(r"`.*?`", "", cleaned)
        cleaned = re.sub(r"^\s*#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"^\s*[\*\-\+]\s+", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
        cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
        cleaned = re.sub(r"_([^_]+)_", r"\1", cleaned)
        cleaned = re.sub(r'["“”]', '', cleaned)
        cleaned = re.sub(r"[\U00010000-\U0010ffff]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if not cleaned:
            cleaned = ATMOSPHERIC_FALLBACK

        state.riddle_response = cleaned

        # Update persistent history
        if state.user_transcript:
            state.conversation_history.append({"role": "user", "text": state.user_transcript})
        state.conversation_history.append({"role": "assistant", "text": cleaned})

        # Structured line segmentation for WebGL Canvas Texture Engine
        line_length_threshold = 48
        words = cleaned.split()
        lines: List[str] = []
        current_line: List[str] = []
        current_len = 0

        for word in words:
            if current_len + len(word) + 1 > line_length_threshold:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_len = len(word)
            else:
                current_line.append(word)
                current_len += len(word) + 1
        if current_line:
            lines.append(" ".join(current_line))

        webgl_texture_payload = {
            "session_id": state.session_id,
            "status": "ink_rendered" if cleaned != ATMOSPHERIC_FALLBACK else "ink_stained",
            "user_transcript": state.user_transcript,
            "riddle_response": cleaned,
            "texture_manifest": {
                "text_content": cleaned,
                "lines": lines,
                "line_count": len(lines),
                "glyph_count": len(cleaned),
                "animation": {
                    "ink_bleed_speed_ms_per_char": 38,
                    "total_render_duration_ms": len(cleaned) * 38,
                    "style": "cursive_quill_fade_in"
                }
            },
            "conversation_history": state.conversation_history
        }

        logger.info("[Node: Clean & Dispatch] Dispatch payload successfully generated.")
        return webgl_texture_payload

    # Orchestrator Pipeline
    async def run(self, state: RiddleDiaryState) -> Dict[str, Any]:
        state = await self.transcribe_audio_node(state)
        state = await self.generate_riddle_reply_node(state)
        return await self.clean_and_dispatch_node(state)


# ---------------------------------------------------------------------------
# 3. FASTAPI NETWORK WORKER WRAPPER
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RiddleDiaryAgent Worker",
    description="Autonomous Horcrux Diary Agent with Voice Ingestion and Three.js Texture Dispatch",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_instance: Optional[RiddleDiaryAgent] = None
session_store: Dict[str, List[Dict[str, str]]] = {}


@app.on_event("startup")
async def startup_event():
    global agent_instance
    agent_instance = RiddleDiaryAgent()


@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "entity": "Tom Riddle Diary Horcrux",
        "whisper_device": getattr(agent_instance, "device", "unknown"),
        "whisper_compute_type": getattr(agent_instance, "compute_type", "unknown"),
        "llm_provider": getattr(agent_instance, "provider", "unknown"),
        "llm_model": getattr(agent_instance, "llm_model", "unknown")
    }


@app.post("/process_voice")
async def process_voice(
    audio_file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    if not agent_instance:
        raise HTTPException(status_code=500, detail="RiddleDiaryAgent is not initialized.")

    active_session = session_id or str(uuid.uuid4())
    history = session_store.get(active_session, [])

    # Keep original audio suffix if provided (e.g. .webm, .wav, .ogg)
    raw_suffix = Path(audio_file.filename).suffix if audio_file.filename else ""
    if not raw_suffix:
        content_type = audio_file.content_type or ""
        if "webm" in content_type:
            raw_suffix = ".webm"
        elif "ogg" in content_type:
            raw_suffix = ".ogg"
        else:
            raw_suffix = ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=raw_suffix) as tmp_file:
        content = await audio_file.read()
        tmp_file.write(content)
        temp_audio_path = tmp_file.name

    try:
        state = RiddleDiaryState(
            session_id=active_session,
            raw_audio_path=temp_audio_path,
            conversation_history=history
        )

        dispatch_payload = await agent_instance.run(state)
        session_store[active_session] = dispatch_payload["conversation_history"]
        return dispatch_payload

    finally:
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except OSError:
                pass


class AudioPathRequest(BaseModel):
    raw_audio_path: str
    session_id: Optional[str] = None


@app.post("/process_audio_path")
async def process_audio_path(payload: AudioPathRequest):
    if not agent_instance:
        raise HTTPException(status_code=500, detail="RiddleDiaryAgent is not initialized.")

    active_session = payload.session_id or str(uuid.uuid4())
    history = session_store.get(active_session, [])

    state = RiddleDiaryState(
        session_id=active_session,
        raw_audio_path=payload.raw_audio_path,
        conversation_history=history
    )

    dispatch_payload = await agent_instance.run(state)
    session_store[active_session] = dispatch_payload["conversation_history"]
    return dispatch_payload


class TextInteractRequest(BaseModel):
    text: str
    session_id: Optional[str] = None


@app.post("/process_text")
async def process_text(payload: TextInteractRequest):
    if not agent_instance:
        raise HTTPException(status_code=500, detail="RiddleDiaryAgent is not initialized.")

    active_session = payload.session_id or str(uuid.uuid4())
    history = session_store.get(active_session, [])

    state = RiddleDiaryState(
        session_id=active_session,
        user_transcript=payload.text,
        conversation_history=history
    )

    state = await agent_instance.generate_riddle_reply_node(state)
    dispatch_payload = await agent_instance.clean_and_dispatch_node(state)
    session_store[active_session] = dispatch_payload["conversation_history"]
    return dispatch_payload


# ---------------------------------------------------------------------------
# SCRIPT ENTRYPOINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    logger.info("Starting RiddleDiaryAgent server on port %d...", port)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
