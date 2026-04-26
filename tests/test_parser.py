import re
from unittest.mock import MagicMock, patch

from src.ingestion.parser import ElementType, ParsedElement, _is_equation, parse_pdf


class TestIsEquation:
    def test_flags_greek_letters(self):
        assert _is_equation("α + β = γ") is True

    def test_flags_math_symbols(self):
        assert _is_equation("∑ x_i = 0") is True

    def test_ignores_long_normal_text(self):
        assert _is_equation("This is a normal sentence about machine learning and results.") is False

    def test_short_text_with_symbols(self):
        assert _is_equation("f(x) = ∫ g(t)dt") is True


class TestParsedElement:
    def test_defaults(self):
        el = ParsedElement(type=ElementType.TEXT, content="hello")
        assert el.metadata == {}
        assert el.image_path is None

    def test_table_type(self):
        el = ParsedElement(type=ElementType.TABLE, content="| a | b |")
        assert el.type == ElementType.TABLE
