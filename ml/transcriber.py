"""
Speech-to-Text Module - Transcription using faster-whisper
"""


class Transcriber:
    """Transcribes audio using faster-whisper."""

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size

    async def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text."""
        raise NotImplementedError
