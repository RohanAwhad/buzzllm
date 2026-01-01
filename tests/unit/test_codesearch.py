import pytest
from pathlib import Path

from buzzllm.tools.codesearch import (
    _validate_path,
    _paginate_results,
    bash_find,
    bash_ripgrep,
    bash_read,
)


class TestValidatePath:
    """Tests for path validation.

    Note: CWD is captured at module import time, so we test
    against the actual project directory.
    """

    def test_valid_relative_path_exists(self):
        # pyproject.toml should exist in the project root
        result = _validate_path("pyproject.toml")
        assert result == "pyproject.toml"

    def test_valid_src_path(self):
        result = _validate_path("src/buzzllm/main.py")
        assert result == "src/buzzllm/main.py"

    def test_valid_dot_path(self):
        result = _validate_path(".")
        assert result == "."

    def test_path_outside_cwd_raises(self):
        with pytest.raises(ValueError, match="Path outside CWD not allowed"):
            _validate_path("/etc/passwd")

    def test_path_traversal_attack_blocked(self):
        with pytest.raises(ValueError, match="Path outside CWD not allowed"):
            _validate_path("../../../etc/passwd")


class TestPaginateResults:
    def test_no_limit_returns_all(self):
        items = ["a", "b", "c", "d", "e"]
        result = _paginate_results(items, limit=0, offset=0)

        assert result["results"] == items
        assert result["total"] == 5
        assert result["returned"] == 5
        assert result["has_more"] is False

    def test_limit_slices_results(self):
        items = ["a", "b", "c", "d", "e"]
        result = _paginate_results(items, limit=2, offset=0)

        assert result["results"] == ["a", "b"]
        assert result["total"] == 5
        assert result["returned"] == 2
        assert result["has_more"] is True

    def test_offset_skips_items(self):
        items = ["a", "b", "c", "d", "e"]
        result = _paginate_results(items, limit=2, offset=2)

        assert result["results"] == ["c", "d"]
        assert result["offset"] == 2
        assert result["has_more"] is True

    def test_offset_beyond_end(self):
        items = ["a", "b", "c"]
        result = _paginate_results(items, limit=10, offset=10)

        assert result["results"] == []
        assert result["returned"] == 0
        assert result["has_more"] is False

    def test_limit_none_returns_from_offset(self):
        items = ["a", "b", "c", "d", "e"]
        result = _paginate_results(items, limit=None, offset=2)

        assert result["results"] == ["c", "d", "e"]
        assert result["has_more"] is False


class TestBashFind:
    """Tests run against actual project directory."""

    def test_finds_files_in_directory(self):
        result = bash_find(path=".", limit=0)

        assert "error" not in result
        assert result["total"] >= 1

    def test_finds_python_files(self):
        result = bash_find(path="src", name="*.py", limit=0)

        assert "error" not in result
        assert result["total"] >= 1
        file_names = result["results"]
        assert all(".py" in f for f in file_names)

    def test_pagination(self):
        result = bash_find(path=".", limit=1, offset=0)

        if result["total"] > 1:
            assert result["returned"] == 1
            assert result["has_more"] is True

    def test_invalid_path_error(self):
        with pytest.raises(ValueError, match="Path outside CWD not allowed"):
            bash_find(path="/etc")

    def test_directory_type_filter(self):
        result = bash_find(path=".", type_filter="d", limit=0)
        # Should find src, tests directories
        assert "error" not in result


class TestBashRipgrep:
    """Tests run against actual project directory."""

    def test_finds_pattern_in_project(self):
        # "def " should be found in Python files
        result = bash_ripgrep(pattern="def ", files="src", limit=0)

        assert "error" not in result
        assert result["total"] >= 1

    def test_no_matches_returns_error(self):
        # Use a pattern that truly doesn't exist (avoid matching this file itself)
        result = bash_ripgrep(pattern="qwerty_asdfgh_zxcvbn_99999", files="src")
        assert "error" in result

    def test_pagination(self):
        result = bash_ripgrep(pattern="import", files="src", limit=2)

        if "error" not in result:
            assert result["returned"] <= 2

    def test_specific_file(self):
        result = bash_ripgrep(pattern="name", files="pyproject.toml", limit=0)

        assert "error" not in result
        assert result["total"] >= 1

    def test_invalid_path_error(self):
        with pytest.raises(ValueError, match="Path outside CWD not allowed"):
            bash_ripgrep(pattern="test", files="/etc/passwd")


class TestBashRead:
    """Tests run against actual project directory."""

    def test_reads_file_contents(self):
        result = bash_read(filepath="pyproject.toml")

        assert "error" not in result
        assert "content" in result
        assert "buzzllm" in result["content"]

    def test_reads_source_file(self):
        result = bash_read(filepath="src/buzzllm/__init__.py")

        assert "error" not in result
        assert "content" in result

    def test_pagination_limit(self):
        result = bash_read(filepath="pyproject.toml", limit=1, offset=0)

        assert result["returned"] == 1
        if result["total"] > 1:
            assert result["has_more"] is True

    def test_file_not_found(self):
        result = bash_read(filepath="nonexistent_file_xyz.py")
        assert "error" in result

    def test_path_outside_cwd(self):
        with pytest.raises(ValueError, match="Path outside CWD not allowed"):
            bash_read(filepath="/etc/passwd")

    def test_returns_line_list_and_content(self):
        result = bash_read(filepath="pyproject.toml", limit=0)

        assert isinstance(result["results"], list)
        assert isinstance(result["content"], str)
        assert result["content"] == "\n".join(result["results"])
