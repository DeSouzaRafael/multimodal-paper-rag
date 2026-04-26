from src.ingestion.chunker import ChunkStrategy, Chunk, chunk_elements
from src.ingestion.parser import ElementType, ParsedElement


def _text_el(content: str, section: str = "intro", page: int = 1) -> ParsedElement:
    return ParsedElement(
        type=ElementType.TEXT,
        content=content,
        metadata={"paper_id": "test", "section": section, "page": page},
    )


def _table_el(content: str) -> ParsedElement:
    return ParsedElement(
        type=ElementType.TABLE,
        content=content,
        metadata={"paper_id": "test", "section": "results", "page": 2},
    )


LONG_TEXT = " ".join([f"word{i}" for i in range(500)])


class TestFixedChunker:
    def test_splits_long_text(self):
        chunks = chunk_elements([_text_el(LONG_TEXT)], strategy=ChunkStrategy.FIXED, chunk_size=100, overlap=10)
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        chunks = chunk_elements([_text_el(LONG_TEXT)], strategy=ChunkStrategy.FIXED, chunk_size=100, overlap=0)
        for c in chunks[:-1]:
            assert len(c.content.split()) <= 100

    def test_overlap_produces_more_chunks(self):
        no_overlap = chunk_elements([_text_el(LONG_TEXT)], strategy=ChunkStrategy.FIXED, chunk_size=100, overlap=0)
        with_overlap = chunk_elements([_text_el(LONG_TEXT)], strategy=ChunkStrategy.FIXED, chunk_size=100, overlap=20)
        assert len(with_overlap) >= len(no_overlap)

    def test_metadata_preserved(self):
        chunks = chunk_elements([_text_el("hello world", section="methods")], strategy=ChunkStrategy.FIXED, chunk_size=10, overlap=0)
        assert all(c.metadata["section"] == "methods" for c in chunks)


class TestRecursiveChunker:
    def test_splits_on_paragraphs_first(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunk_elements([_text_el(text)], strategy=ChunkStrategy.RECURSIVE, chunk_size=10, overlap=0)
        assert len(chunks) >= 3

    def test_short_text_single_chunk(self):
        chunks = chunk_elements([_text_el("short text")], strategy=ChunkStrategy.RECURSIVE, chunk_size=100, overlap=0)
        assert len(chunks) == 1


class TestTablePassThrough:
    def test_table_not_split(self):
        table_content = "| col1 | col2 |\n|------|------|\n| a    | b    |"
        chunks = chunk_elements([_table_el(table_content)], strategy=ChunkStrategy.FIXED, chunk_size=5, overlap=0)
        assert len(chunks) == 1
        assert chunks[0].element_type == ElementType.TABLE
        assert chunks[0].content == table_content


class TestChunkMetadata:
    def test_chunk_index_added(self):
        chunks = chunk_elements([_text_el(LONG_TEXT)], strategy=ChunkStrategy.FIXED, chunk_size=50, overlap=0)
        indices = [c.metadata.get("chunk_index") for c in chunks if c.element_type == ElementType.TEXT]
        assert indices == list(range(len(indices)))

    def test_strategy_recorded_in_metadata(self):
        chunks = chunk_elements([_text_el("some text")], strategy=ChunkStrategy.RECURSIVE)
        assert all(c.metadata.get("strategy") == "recursive" for c in chunks)
