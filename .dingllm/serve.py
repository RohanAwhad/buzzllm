"""
Lightweight doc server for .dingllm/ directory.
Renders .md files with marked.js and .mmd files with Mermaid.js (both via CDN).

Usage: python serve.py [--port 8000]
"""

import http.server
import json
import argparse
import os
import sys
import threading
import time
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).parent

_file_version = 0
_last_mtimes: dict[str, float] = {}


def _scan_mtimes() -> dict[str, float]:
    mtimes = {}
    for p in ROOT.rglob("*"):
        if p.suffix in (".md", ".mmd", ".html") and p.is_file():
            mtimes[str(p)] = p.stat().st_mtime
    return mtimes


def _watch_files():
    global _file_version, _last_mtimes
    _last_mtimes = _scan_mtimes()
    while True:
        time.sleep(1)
        current = _scan_mtimes()
        if current != _last_mtimes:
            _last_mtimes = current
            _file_version += 1


def _watch_self():
    mtime = Path(__file__).stat().st_mtime
    while True:
        time.sleep(1)
        current = Path(__file__).stat().st_mtime
        if current != mtime:
            print("  serve.py changed, restarting...")
            os.execv(sys.executable, [sys.executable] + sys.argv)


HTML_SHELL = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; max-width: 1400px; margin: 2rem auto; padding: 0 1rem; color: #1f2328; background: #fff; }}
  a {{ color: #0969da; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  pre {{ background: #f6f8fa; padding: 1rem; border-radius: 6px; overflow-x: auto; }}
  code {{ background: #f6f8fa; padding: 0.2em 0.4em; border-radius: 3px; font-size: 85%; }}
  pre code {{ background: none; padding: 0; }}
  .nav {{ margin-bottom: 1.5rem; padding-bottom: 0.5rem; border-bottom: 1px solid #d1d9e0; }}
  .file-list {{ list-style: none; padding: 0; }}
  .file-list li {{ padding: 0.4rem 0; }}
  .file-list .dir {{ font-weight: 600; margin-top: 1rem; }}
  .mermaid {{ display: flex; justify-content: center; margin: 2rem 0; }}
  table {{ border-collapse: collapse; }}
  th, td {{ border: 1px solid #d1d9e0; padding: 0.5rem 1rem; }}
</style>
</head><body>
<div class="nav"><a href="/">index</a></div>
{body}
{scripts}
<script>
(function() {{
  var es = new EventSource('/_events');
  var connected = false;
  es.onopen = function() {{
    if (connected) location.reload();
    connected = true;
  }};
  es.onmessage = function(e) {{ if (e.data === 'reload') location.reload(); }};
}})();
</script>
</body></html>"""

MARKED_SCRIPTS = """
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>
  const src = document.getElementById('md-source').textContent;
  document.getElementById('content').innerHTML = marked.parse(src);
</script>
"""

MERMAID_SCRIPTS = """
<script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.2/dist/svg-pan-zoom.min.js"></script>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: false, theme: 'default' });

  const src = document.getElementById('mmd-source').textContent;
  const preview = document.getElementById('mmd-preview');
  try {
    const { svg } = await mermaid.render('mmd-output', src);
    preview.innerHTML = svg;
    const svgEl = preview.querySelector('svg');
    if (svgEl) {
      svgEl.setAttribute('width', '100%');
      svgEl.setAttribute('height', '100%');
      svgEl.style.maxWidth = '100%';
      svgPanZoom(svgEl, {
        center: true,
        fit: true,
        controlIconsEnabled: false,
        zoomEnabled: true,
        panEnabled: true,
        minZoom: 0.2,
        maxZoom: 12,
        zoomScaleSensitivity: 0.3,
      });
    }
  } catch (e) {
    preview.innerHTML = '<pre style="color:#cf222e;padding:1rem;">' + e.message + '</pre>';
  }
</script>
"""


def collect_files():
    """Walk ROOT and return a dict of {relative_dir: [filenames]} for .md and .mmd files."""
    tree = {}
    for path in sorted(ROOT.rglob("*")):
        if path.suffix not in (".md", ".mmd", ".html"):
            continue
        if path.name == "serve.py":
            continue
        rel = path.relative_to(ROOT)
        parent = str(rel.parent) if rel.parent != Path(".") else ""
        tree.setdefault(parent, []).append(rel)
    return tree


def _file_list_html(active: Path | None = None) -> str:
    tree = collect_files()
    all_dirs = sorted(tree.keys())

    def _child_dirs(prefix: str) -> list[str]:
        if prefix == "":
            return sorted({d.split("/")[0] for d in all_dirs if d})
        return sorted({
            d for d in all_dirs
            if d.startswith(prefix + "/") and d.count("/") == prefix.count("/") + 1
        })

    def _has_active(prefix: str) -> bool:
        """Check if active file is anywhere under this prefix."""
        if not active:
            return False
        for d in all_dirs:
            if d == prefix or d.startswith(prefix + "/"):
                if any(r == active for r in tree.get(d, [])):
                    return True
        return False

    def _render_dir(prefix: str) -> str:
        lines: list[str] = []
        for rel_path in sorted(tree.get(prefix, [])):
            suffix_label = {".mmd": "mermaid", ".md": "md", ".html": "html"}.get(rel_path.suffix, rel_path.suffix)
            name = f"<strong>{rel_path.name}</strong>" if active and rel_path == active else rel_path.name
            lines.append(f'<li><a href="/{rel_path}">{name}</a> <small>({suffix_label})</small></li>')
        for child in _child_dirs(prefix):
            dir_label = child.rsplit("/", 1)[-1]
            open_attr = " open" if _has_active(child) else ""
            inner = _render_dir(child)
            lines.append(
                f'<li><details{open_attr}><summary class="dir">{dir_label}/</summary>'
                f'<ul class="file-list">{inner}</ul></details></li>'
            )
        return "\n".join(lines)

    return "<ul class='file-list'>\n" + _render_dir("") + "\n</ul>"


def render_index():
    body = "<h1>.dingllm docs</h1>\n" + _file_list_html()
    return HTML_SHELL.format(title=".dingllm", body=body, scripts="")


def render_md(path: Path):
    content = path.read_text()
    # Escape for embedding in a hidden <script> tag
    escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    body = f'<script type="text/plain" id="md-source">{escaped}</script>\n<div id="content"></div>'
    return HTML_SHELL.format(title=path.name, body=body, scripts=MARKED_SCRIPTS)


def render_html(path: Path):
    rel = path.relative_to(ROOT)
    index_html = _file_list_html(active=rel)
    body = f"""
<div style="width:80vw;margin-left:calc(50% - 40vw);height:calc(100vh - 8rem);display:flex;gap:1.5rem;">
  <div style="flex:0 0 250px;overflow-y:auto;padding-right:1rem;border-right:1px solid #d1d9e0;">
    <h3 style="margin-top:0;">.dingllm</h3>
    {index_html}
  </div>
  <iframe src="/_raw/{rel}" style="flex:1;border:1px solid #d1d9e0;border-radius:6px;" frameborder="0"></iframe>
</div>"""
    return HTML_SHELL.format(title=path.name, body=body, scripts="")


def render_mmd(path: Path):
    content = path.read_text()
    rel = path.relative_to(ROOT)
    index_html = _file_list_html(active=rel)
    body = f"""
<script type="text/plain" id="mmd-source">{content}</script>
<div style="width:80vw;margin-left:calc(50% - 40vw);height:calc(100vh - 8rem);display:flex;gap:1.5rem;">
  <div style="flex:0 0 250px;overflow-y:auto;padding-right:1rem;border-right:1px solid #d1d9e0;">
    <h3 style="margin-top:0;">.dingllm</h3>
    {index_html}
  </div>
  <div id="mmd-preview" style="flex:1;overflow:hidden;border:1px solid #d1d9e0;border-radius:6px;padding:1rem;position:relative;"></div>
</div>"""
    return HTML_SHELL.format(title=path.name, body=body, scripts=MERMAID_SCRIPTS)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = unquote(self.path).lstrip("/")

        if path == "" or path == "/":
            self._respond(200, render_index())
            return

        if path == "_events":
            self._handle_sse()
            return

        if path.startswith("_raw/"):
            raw_path = ROOT / path[5:]
            if raw_path.exists() and raw_path.is_file() and raw_path.suffix == ".html":
                self._respond(200, raw_path.read_text())
            else:
                self._respond(404, "<h1>404</h1>")
            return

        file_path = ROOT / path
        if not file_path.exists() or not file_path.is_file():
            self._respond(404, HTML_SHELL.format(title="404", body="<h1>404</h1><p>Not found.</p>", scripts=""))
            return

        if file_path.suffix == ".md":
            self._respond(200, render_md(file_path))
        elif file_path.suffix == ".mmd":
            self._respond(200, render_mmd(file_path))
        elif file_path.suffix == ".html":
            self._respond(200, render_html(file_path))
        elif file_path.suffix == ".png":
            self._respond_binary(200, file_path.read_bytes(), "image/png")
        else:
            self._respond(404, HTML_SHELL.format(title="404", body="<h1>404</h1><p>Unsupported file type.</p>", scripts=""))

    def _handle_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        last_seen = _file_version
        while True:
            time.sleep(1)
            if _file_version != last_seen:
                last_seen = _file_version
                self.wfile.write(b"data: reload\n\n")
                self.wfile.flush()

    def do_POST(self):
        path = unquote(self.path).lstrip("/")
        file_path = ROOT / path

        if not file_path.exists() or file_path.suffix != ".mmd":
            self._respond_json(400, {"error": "can only save .mmd files"})
            return

        # Ensure we're not writing outside ROOT
        if ROOT not in file_path.resolve().parents and file_path.resolve() != ROOT:
            self._respond_json(403, {"error": "forbidden"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        file_path.write_text(body)
        _last_mtimes[str(file_path)] = file_path.stat().st_mtime
        print(f"  saved {file_path.relative_to(ROOT)}")
        self._respond_json(200, {"ok": True})

    def _respond_json(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _respond(self, code, html):
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())

    def _respond_binary(self, code, data, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        print(f"  {args[0]}")


def main():
    parser = argparse.ArgumentParser(description="Serve .dingllm docs")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    threading.Thread(target=_watch_files, daemon=True).start()
    threading.Thread(target=_watch_self, daemon=True).start()

    server = http.server.ThreadingHTTPServer(("", args.port), Handler)
    print(f"Serving .dingllm at http://localhost:{args.port} (hot reload enabled)")
    server.serve_forever()


if __name__ == "__main__":
    main()
