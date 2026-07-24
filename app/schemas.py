"""Core data contracts for the extraction -> HITL -> ingestion pipeline.

The confidence + bbox schema is the backbone of the product: every extracted
field carries its provenance (bbox on the source page) and a confidence level,
and illegible values must be null rather than guessed.
"""
from enum import Enum

from pydantic import BaseModel


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class VerificationStatus(str, Enum):
    unverified = "unverified"
    confirmed = "confirmed"
    corrected = "corrected"


class RegionType(str, Enum):
    summary = "summary"  # whole-drawing description written by the vision model
    title_block = "title_block"
    dimension = "dimension"
    note = "note"
    bom = "bom"
    view = "view"
    # a drawn physical element (stair, pipe, valve, pump, door, equipment) -
    # located visually by the vision model, not read as text
    component = "component"


class ProvisionalChunk(BaseModel):
    """One extracted region awaiting HITL review, before ingestion."""

    region_type: RegionType
    chunk_text: str | None  # null when the source value was unreadable
    bbox: list[float] | None = None
    confidence: Confidence = Confidence.high
    page: int = 1
    # vision verdict, set on the summary chunk only: False when the image is
    # not an engineering drawing (photo, screenshot, ...); None = not judged
    is_drawing: bool | None = None
    # advisory chunks are pipeline disclosures (e.g. "converted from DWG,
    # accuracy may be affected") - shown as a banner in the UI, never
    # ingested into the knowledge base
    advisory: bool = False
    # component regions: every further instance of the same component type
    # (bbox holds the first). One card per type, all locations highlightable.
    extra_bboxes: list[list[float]] | None = None


class ExtractedField(BaseModel):
    """A single extracted value with provenance and confidence."""

    value: str | None  # null when illegible/ambiguous — never guessed
    confidence: Confidence
    bbox: tuple[float, float, float, float] | None  # x1, y1, x2, y2
    status: VerificationStatus = VerificationStatus.unverified
    corrected_value: str | None = None


class Dimension(ExtractedField):
    unit: str | None = None


class TitleBlock(BaseModel):
    drawing_number: ExtractedField | None = None
    revision: ExtractedField | None = None
    title: ExtractedField | None = None


class BomRow(BaseModel):
    item: ExtractedField | None = None
    part_number: ExtractedField | None = None
    description: ExtractedField | None = None
    quantity: ExtractedField | None = None


class ExtractionResult(BaseModel):
    """Full structured output of the extraction step for one page."""

    title_block: TitleBlock | None = None
    dimensions: list[Dimension] = []
    notes: list[ExtractedField] = []
    bom: list[BomRow] = []


class Chunk(BaseModel):
    """One semantic region of a drawing, ready for embedding + storage."""

    source_file_id: str
    page: int
    region_type: RegionType
    chunk_text: str
    bbox: tuple[float, float, float, float] | None = None
    image_uri: str | None = None  # S3 pointer to the evidence crop
    confidence: Confidence = Confidence.high
    verification_status: VerificationStatus = VerificationStatus.unverified


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class Evidence(BaseModel):
    source_file_id: str
    page: int
    bbox: tuple[float, float, float, float] | None
    image_uri: str | None
    chunk_text: str


class QueryResponse(BaseModel):
    answer: str
    evidence: list[Evidence]
