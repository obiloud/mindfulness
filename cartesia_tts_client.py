import os
from typing import Generator, Iterable, Literal, Optional

from cartesia import Cartesia

from constants import RATE
from utils import parse_pause_tags, recursive_word_chunker


VoiceMode = Literal["id", "design"]


class CartesiaTTSClient:
    """
    Thin wrapper around Cartesia's Python SDK.

    This client:
    - interprets [PAUSE:x] tags in the transcript
    - chunks long text segments using the existing chunker
    - yields raw audio bytes suitable for streaming or saving.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "sonic-3",
        voice_mode: VoiceMode = "design",
        voice_id: Optional[str] = None,
        sample_rate: int = RATE,
    ) -> None:
        self.client = Cartesia(api_key=api_key or os.getenv("CARTESIA_API_KEY"))
        self.model_id = model_id
        self.voice_mode = voice_mode
        self.voice_id = voice_id
        self.sample_rate = sample_rate

    def _voice_payload(self, voice_character: Optional[str] = None):
        """
        Build the `voice` payload for Cartesia.

        For now, we use design mode and pass the free-form description.
        If `voice_mode == 'id'` and `voice_id` is set, we use a fixed voice.
        """
        if self.voice_mode == "id" and self.voice_id:
            return {"mode": "id", "id": self.voice_id}

        # Default: design mode using the natural language description.
        return {
            "mode": "design",
            "text": voice_character
            or "Realistic, calm, warm, slow-paced meditation voice.",
        }

    def _iter_tts_bytes(self, text: str, voice_character: Optional[str] = None) -> Iterable[bytes]:
        voice = self._voice_payload(voice_character)
        output_format = {
            "container": "wav",
            "sample_rate": self.sample_rate,
            "encoding": "pcm_s16le",
        }

        return self.client.tts.bytes(
            model_id=self.model_id,
            transcript=text,
            voice=voice,
            output_format=output_format,
        )

    def stream_bytes(self, transcript: str, voice_character: Optional[str] = None, max_words_per_chunk: int = 60) -> Generator[bytes, None, None]:
        """
        Interpret pauses, chunk text segments, and stream Cartesia audio bytes.
        """
        sequence = parse_pause_tags(transcript)

        for item in sequence:
            # Pauses are best handled on the client side (player) or by inserting silence.
            if isinstance(item, float):
                # Caller can translate this into silence if desired.
                # We just signal with an empty chunk and duration semantics can be
                # handled at a higher layer if needed.
                continue

            if isinstance(item, str) and item.strip():
                for chunk in recursive_word_chunker(item, max_words_per_chunk):
                    for audio_bytes in self._iter_tts_bytes(chunk, voice_character):
                        yield audio_bytes

