import inspect
from typing import Callable, get_type_hints, get_origin

AVAILABLE_TOOLS = {}


def add_tool(func: Callable):
    AVAILABLE_TOOLS[func.__name__] = func


def callable_to_openai_schema(func, desc: str = ""):
    name = func.__name__
    description = desc or inspect.getdoc(func)
    if not description:
        raise ValueError(
            f"neither docstring or description is provided for function: '{name}'"
        )

    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        param_type = type_hints.get(param_name, str)
        properties[param_name] = {"type": _python_type_to_json_type(param_type)}

        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def callable_to_anthropic_schema(func, desc: str = ""):
    name = func.__name__
    description = desc or inspect.getdoc(func)
    if not description:
        raise ValueError(
            f"neither docstring or description is provided for function: '{name}'"
        )

    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        param_type = type_hints.get(param_name, str)
        properties[param_name] = {"type": _python_type_to_json_type(param_type)}

        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def _python_type_to_json_type(python_type):
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    if python_type in type_map:
        return type_map[python_type]

    origin = get_origin(python_type)
    if origin in type_map:
        return type_map[origin]

    return "string"  # fallback
