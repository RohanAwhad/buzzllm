from . import websearch, codesearch, pythonexec


TOOL_CATALOG = {
    "search_web": {
        "callable": websearch.search_web,
        "desc": "",
    },
    "scrape_webpage": {
        "callable": websearch.scrape_webpage,
        "desc": "",
    },
    "bash_find": {
        "callable": codesearch.bash_find,
        "desc": codesearch.bash_find_tool_desc,
    },
    "bash_ripgrep": {
        "callable": codesearch.bash_ripgrep,
        "desc": codesearch.bash_ripgrep_tool_desc,
    },
    "bash_read": {
        "callable": codesearch.bash_read,
        "desc": "",
    },
    "python_execute": {
        "callable": pythonexec.python_execute,
        "desc": "",
    },
}


TOOL_NAMES = set(TOOL_CATALOG.keys())
