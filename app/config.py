from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    chat_model: str
    max_upload_bytes: int
    max_document_chars: int
    min_similarity: float

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            data_dir=Path(os.getenv("DATA_DIR", "./data")).resolve(),
            chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", "5242880")),
            max_document_chars=int(os.getenv("MAX_DOCUMENT_CHARS", "100000")),
            min_similarity=float(os.getenv("MIN_SIMILARITY", "0.30")),
        )
