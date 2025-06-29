import inspect
from typing import Callable, get_type_hints, get_origin, get_args, Union

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
        properties[param_name] = _python_type_to_json_schema(param_type)
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
        properties[param_name] = _python_type_to_json_schema(param_type)
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


def _python_type_to_json_schema(python_type):
    """
    Return a JSON Schema fragment for the given Python typing object.
    Supports builtin primitives, List[T], Dict[K, V], Optional[T], and Union[…].
    """
    primitive_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    # Support NoneType
    if python_type is type(None):
        return {"type": "null"}

    origin = get_origin(python_type)
    args = get_args(python_type)

    # Handle Union (including Optional)
    if origin is Union:
        # flatten nested Unions
        branches = []
        for arg in args:
            if get_origin(arg) is Union:
                branches.extend(get_args(arg))
            else:
                branches.append(arg)

        subschemas = [_python_type_to_json_schema(b) for b in branches]
        # if all branches are primitive types, collapse into a type array
        primitive_types = []
        for s in subschemas:
            t = s.get("type")
            if isinstance(t, str):
                primitive_types.append(t)
            else:
                primitive_types = None
                break
        if primitive_types is not None:
            return {"type": primitive_types}
        # otherwise use oneOf
        return {"oneOf": subschemas}

    # Simple builtins
    if python_type in primitive_map:
        return {"type": primitive_map[python_type]}

    # Parameterized containers
    if origin is list:
        item_schema = _python_type_to_json_schema(args[0]) if args else {}
        return {"type": "array", "items": item_schema}

    if origin is dict:
        # JSON Schema objects only define value types
        value_schema = _python_type_to_json_schema(args[1]) if len(args) > 1 else {}
        return {"type": "object", "additionalProperties": value_schema}

    raise NotImplementedError(f"decoding of '{python_type}' is not yet implemented")
