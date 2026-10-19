"""A tiny local web server (Python standard library only) for the interactive demo.

    make demo            # then open http://127.0.0.1:8770

GET  /               the page
GET  /api/info       what is loaded (probes, voices, speakers)
POST /api/run        {"kind": "pitch"|"highband"|"distance", "params": {...}}  -> clip + per-layer predictions
POST /api/sweep      {"params": {...}, "distances": [...]}                      -> predictions at each distance
POST /api/file       {"path": "/System/Library/Sounds/Glass.aiff"} or {"b64": "...", "name": "x.m4a"}
"""

from __future__ import annotations

import argparse
import base64
import json
import tempfile
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from audio_probes.demo.lab import Lab
from audio_probes.predict import decode_file

STATIC = Path(__file__).parent / "static"
SYSTEM_SOUNDS = Path("/System/Library/Sounds")


def make_handler(lab: Lab):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the terminal quiet except for errors
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/info":
                sounds = sorted(str(p) for p in SYSTEM_SOUNDS.glob("*.aiff")) if SYSTEM_SOUNDS.exists() else []
                self._json({**lab.info(), "system_sounds": sounds})
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/api/run":
                    self._json(lab.run(body["kind"], body["params"]))
                elif self.path == "/api/sweep":
                    self._json(lab.sweep(body["params"], body["distances"]))
                elif self.path == "/api/file":
                    self._json(lab.run_file(load_audio(body), body.get("condition", "raw")))
                else:
                    self._send(404, b"not found", "text/plain")
            except ValueError as err:  # a user-fixable problem, e.g. the distance does not fit the room
                self._json({"error": str(err)}, 400)
            except Exception as err:
                traceback.print_exc()
                self._json({"error": f"{type(err).__name__}: {err}"}, 500)

    return Handler


def load_audio(body: dict):
    """A system sound by path, or an uploaded file (base64), decoded by ffmpeg to 16 kHz mono."""
    if "path" in body:
        path = Path(body["path"]).resolve()
        if path.parent != SYSTEM_SOUNDS:
            raise ValueError("Only files in /System/Library/Sounds can be opened by path; upload anything else.")
        return decode_file(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(body.get("name", "upload")).name
        path.write_bytes(base64.b64decode(body["b64"]))
        return decode_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive audio-probe demo on localhost")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    print("[demo] loading Whisper and the trained probes (about 1.5 GB of memory while running)...", flush=True)
    lab = Lab()
    url = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(lab))  # localhost only
    print(f"[demo] ready on {url}  (device={lab.encoder.device}; Ctrl+C to stop)", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[demo] stopped", flush=True)
