import pytest

from buzzllm.tools import utils


class TestBuildToolSchemas:
    def test_build_tool_schemas_with_valid_subset(self):
        schemas = utils.build_tool_schemas(
            ["search_web", "bash_find"], utils.callable_to_openai_schema
        )

        assert len(schemas) == 2
        assert schemas[0]["function"]["name"] == "search_web"
        assert schemas[1]["function"]["name"] == "bash_find"
        assert "search_web" in utils.AVAILABLE_TOOLS
        assert "bash_find" in utils.AVAILABLE_TOOLS

    def test_build_tool_schemas_rejects_unknown_tool(self):
        with pytest.raises(ValueError, match="Unknown tool name"):
            utils.build_tool_schemas(["unknown_tool"], utils.callable_to_openai_schema)
