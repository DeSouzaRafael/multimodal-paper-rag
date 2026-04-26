from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from unstructured.documents.elements import (
    Image,
    NarrativeText,
    Table,
    Text,
    Title,
)
from unstructured.partition.pdf import partition_pdf


class ElementType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


@dataclass
class ParsedElement:
    type: ElementType
    content: str
    metadata: dict = field(default_factory=dict)
    image_path: str | None = None


_EQUATION_RE = re.compile(r"[\\∑∫∂∇αβγδεζηθλμνξπρστφχψω=]")
_CAPTION_RE = re.compile(r"(?i)^fig(ure)?[.\s]\s*\d")


def _is_equation(text: str) -> bool:
    return bool(_EQUATION_RE.search(text)) and len(text) < 120


def _find_caption(elements: list, image_page: int) -> str:
    for el in elements:
        if el.metadata.page_number == image_page and _CAPTION_RE.match(str(el)):
            return str(el)
    return ""


def parse_pdf(path: str | Path, extract_images: bool = True) -> list[ParsedElement]:
    path = Path(path)
    paper_id = path.stem

    raw = partition_pdf(
        filename=str(path),
        strategy="hi_res",
        extract_images_in_pdf=extract_images,
        infer_table_structure=True,
        include_page_breaks=False,
    )

    results: list[ParsedElement] = []
    section = "unknown"

    for el in raw:
        page = getattr(el.metadata, "page_number", None)
        text = str(el).strip()

        if not text:
            continue

        if isinstance(el, Title):
            section = text
            results.append(ParsedElement(
                type=ElementType.TEXT,
                content=text,
                metadata={"paper_id": paper_id, "section": section, "page": page, "is_title": True},
            ))

        elif isinstance(el, Table):
            html = getattr(el.metadata, "text_as_html", None) or text
            results.append(ParsedElement(
                type=ElementType.TABLE,
                content=html,
                metadata={"paper_id": paper_id, "section": section, "page": page},
            ))

        elif isinstance(el, Image):
            caption = _find_caption(raw, page)
            img_path = getattr(el.metadata, "image_path", None)
            results.append(ParsedElement(
                type=ElementType.IMAGE,
                content=caption or text,
                metadata={"paper_id": paper_id, "section": section, "page": page, "caption": caption},
                image_path=img_path,
            ))

        elif isinstance(el, (NarrativeText, Text)):
            if _is_equation(text):
                continue
            results.append(ParsedElement(
                type=ElementType.TEXT,
                content=text,
                metadata={"paper_id": paper_id, "section": section, "page": page},
            ))

    return results
