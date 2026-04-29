"""Audio captcha solver using Whisper (local) and Vapi (cloud fallback)."""

import io
import os
import logging
import tempfile
from pathlib import Path

import httpx

from config import config

logger = logging.getLogger(__name__)


class AudioEngine:
    """Solves audio captchas via speech-to-text: Whisper local model + Vapi API fallback."""

    def __init__(self):
        self._whisper_model = None

    def _load_whisper(self):
        if self._whisper_model is None:
            try:
                import whisper
                self._whisper_model = whisper.load_model(config.whisper_model)
                logger.info(f"Whisper model '{config.whisper_model}' loaded")
            except Exception as e:
                logger.error(f"Failed to load Whisper: {e}")
        return self._whisper_model

    async def _download_audio(self, url: str) -> bytes | None:
        """Download audio file from URL."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.content
        except Exception as e:
            logger.error(f"Failed to download audio from {url}: {e}")
            return None

    def _transcribe_whisper(self, audio_data: bytes) -> str:
        """Transcribe audio using local Whisper model."""
        model = self._load_whisper()
        if model is None:
            return ""

        # Write to temp file (Whisper needs file path)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_data)
            tmp_path = f.name

        try:
            result = model.transcribe(tmp_path, language="en")
            text = result.get("text", "").strip()
            # Clean: extract only digits/letters (captcha audio is usually digits)
            cleaned = "".join(c for c in text if c.isalnum() or c == " ")
            logger.info(f"Whisper transcription: '{cleaned}'")
            return cleaned
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return ""
        finally:
            os.unlink(tmp_path)

    async def _transcribe_vapi(self, audio_data: bytes) -> str:
        """Transcribe audio using Vapi API (cloud fallback)."""
        if not config.vapi_api_key:
            logger.debug("No Vapi API key configured, skipping")
            return ""

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{config.vapi_base_url}/transcription",
                    headers={
                        "Authorization": f"Bearer {config.vapi_api_key}",
                        "Content-Type": "audio/mpeg",
                    },
                    content=audio_data,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data.get("text", data.get("transcript", ""))
                logger.info(f"Vapi transcription: '{text}'")
                return text.strip()
        except Exception as e:
            logger.error(f"Vapi transcription failed: {e}")
            return ""

    async def solve(self, captcha_type=None, pageurl="", sitekey=None,
                    image_data=None, image_url=None, extra=None) -> dict:
        """
        Solve an audio captcha challenge.

        Expects either:
        - extra.audio_url: URL to the audio challenge MP3
        - extra.audio_data: Raw audio bytes
        """
        extra = extra or {}
        audio_url = extra.get("audio_url", "")
        audio_data = extra.get("audio_data", None)

        if audio_url and not audio_data:
            audio_data = await self._download_audio(audio_url)

        if not audio_data:
            return {"success": False, "error": "No audio data provided"}

        if isinstance(audio_data, str):
            import base64
            audio_data = base64.b64decode(audio_data)

        # Try Whisper first (local, free)
        text = self._transcribe_whisper(audio_data)

        # Fallback to Vapi if Whisper fails
        if not text:
            text = await self._transcribe_vapi(audio_data)

        if not text:
            return {"success": False, "error": "Transcription failed on all engines"}

        # Extract digits if this is a digit-based audio captcha
        digits_only = "".join(c for c in text if c.isdigit())
        # For reCAPTCHA audio, the response is usually spoken words/numbers
        response = digits_only if digits_only else text

        return {
            "success": True,
            "token": response,
            "confidence": 0.80,
            "raw_transcription": text,
        }
