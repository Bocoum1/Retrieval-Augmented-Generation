from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class DocumentUnit:
    text: str
    source: str
    page: Optional[int]
    doc_type: str


@dataclass
class Chunk:
    chunk_id: str
    content: str
    source: str
    page: Optional[int]
    doc_type: str
    position: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)