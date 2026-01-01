import pytest
from typing import Optional, Union, List, Dict

from buzzllm.tools.utils import (
    add_tool,
    callable_to_openai_schema,
    callable_to_anthropic_schema,
    _python_type_to_json_schema,
    AVAILABLE_TOOLS,
)


class TestAddTool:
    def test_add_tool_registers_function(self):
        def my_tool():
            """A tool"""
            pass

        add_tool(my_tool)
        assert "my_tool" in AVAILABLE_TOOLS
        assert AVAILABLE_TOOLS["my_tool"] is my_tool

    def test_add_tool_overwrites_existing(self):
        def my_tool():
            """First version"""
            pass

        def my_tool():  # noqa: F811
            """Second version"""
            pass

        add_tool(my_tool)
        assert AVAILABLE_TOOLS["my_tool"].__doc__ == "Second version"


class TestCallableToOpenaiSchema:
    def test_basic_function_with_docstring(self, sample_function_with_docstring):
        schema = callable_to_openai_schema(sample_function_with_docstring)

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "my_func"
        assert "sample function" in schema["function"]["description"]
        assert schema["function"]["parameters"]["type"] == "object"
        assert "name" in schema["function"]["parameters"]["properties"]
        assert "count" in schema["function"]["parameters"]["properties"]
        assert schema["function"]["parameters"]["required"] == ["name"]

    def test_function_without_docstring_raises(self, sample_function_without_docstring):
        with pytest.raises(ValueError, match="neither docstring"):
            callable_to_openai_schema(sample_function_without_docstring)

    def test_custom_description_overrides_docstring(self, sample_function_with_docstring):
        schema = callable_to_openai_schema(
            sample_function_with_docstring, desc="Custom description"
        )
        assert schema["function"]["description"] == "Custom description"

    def test_type_conversion_str(self):
        def func(x: str):
            """Doc"""
            pass

        schema = callable_to_openai_schema(func)
        assert schema["function"]["parameters"]["properties"]["x"]["type"] == "string"

    def test_type_conversion_int(self):
        def func(x: int):
            """Doc"""
            pass

        schema = callable_to_openai_schema(func)
        assert schema["function"]["parameters"]["properties"]["x"]["type"] == "integer"

    def test_type_conversion_float(self):
        def func(x: float):
            """Doc"""
            pass

        schema = callable_to_openai_schema(func)
        assert schema["function"]["parameters"]["properties"]["x"]["type"] == "number"

    def test_type_conversion_bool(self):
        def func(x: bool):
            """Doc"""
            pass

        schema = callable_to_openai_schema(func)
        assert schema["function"]["parameters"]["properties"]["x"]["type"] == "boolean"

    def test_optional_param_not_required(self):
        def func(x: str, y: int = 5):
            """Doc"""
            pass

        schema = callable_to_openai_schema(func)
        assert "x" in schema["function"]["parameters"]["required"]
        assert "y" not in schema["function"]["parameters"]["required"]


class TestCallableToAnthropicSchema:
    def test_basic_function_with_docstring(self, sample_function_with_docstring):
        schema = callable_to_anthropic_schema(sample_function_with_docstring)

        assert schema["name"] == "my_func"
        assert "sample function" in schema["description"]
        assert schema["input_schema"]["type"] == "object"
        assert "name" in schema["input_schema"]["properties"]
        assert schema["input_schema"]["required"] == ["name"]

    def test_function_without_docstring_raises(self, sample_function_without_docstring):
        with pytest.raises(ValueError, match="neither docstring"):
            callable_to_anthropic_schema(sample_function_without_docstring)

    def test_has_input_schema_not_parameters(self, sample_function_with_docstring):
        schema = callable_to_anthropic_schema(sample_function_with_docstring)
        assert "input_schema" in schema
        assert "parameters" not in schema


class TestPythonTypeToJsonSchema:
    def test_str(self):
        assert _python_type_to_json_schema(str) == {"type": "string"}

    def test_int(self):
        assert _python_type_to_json_schema(int) == {"type": "integer"}

    def test_float(self):
        assert _python_type_to_json_schema(float) == {"type": "number"}

    def test_bool(self):
        assert _python_type_to_json_schema(bool) == {"type": "boolean"}

    def test_list_unparameterized(self):
        assert _python_type_to_json_schema(list) == {"type": "array"}

    def test_dict_unparameterized(self):
        assert _python_type_to_json_schema(dict) == {"type": "object"}

    def test_list_parameterized(self):
        result = _python_type_to_json_schema(List[str])
        assert result == {"type": "array", "items": {"type": "string"}}

    def test_dict_parameterized(self):
        result = _python_type_to_json_schema(Dict[str, int])
        assert result == {"type": "object", "additionalProperties": {"type": "integer"}}

    def test_optional_type(self):
        result = _python_type_to_json_schema(Optional[str])
        assert result == {"type": ["string", "null"]}

    def test_union_type(self):
        result = _python_type_to_json_schema(Union[str, int])
        assert result == {"type": ["string", "integer"]}

    def test_none_type(self):
        result = _python_type_to_json_schema(type(None))
        assert result == {"type": "null"}

    def test_unsupported_type_raises(self):
        class CustomClass:
            pass

        with pytest.raises(NotImplementedError):
            _python_type_to_json_schema(CustomClass)
