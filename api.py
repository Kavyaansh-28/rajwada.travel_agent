"""Small JSON API for the React client."""
from __future__ import annotations

import json
import errno
import os
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import URLError
from urllib.request import urlopen

from agent import RajasthanAgent
from tools import get_transport_options

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_ORIGINS = {"http://localhost:8443", "http://127.0.0.1:8443", "http://localhost:5173", "http://127.0.0.1:5173"}


class ApiHandler(BaseHTTPRequestHandler):
    """Adapt the bounded planner to a browser-friendly JSON endpoint."""

    def _send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        origin = self.headers.get("Origin")
        configured = {x.strip() for x in os.getenv("CORS_ORIGINS", "").split(",") if x.strip()}
        allowed = configured or DEFAULT_ORIGINS
        if origin in allowed:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        origin = self.headers.get("Origin")
        configured = {x.strip() for x in os.getenv("CORS_ORIGINS", "").split(",") if x.strip()}
        allowed = configured or DEFAULT_ORIGINS
        if origin in allowed:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/") == "/api/health":
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").lower()
            provider = "Groq" if "api.groq.com" in base_url else "OpenAI-compatible" if "api.openai.com" not in base_url else "OpenAI"
            self._send_json({"status": "ok", "llm_enabled": bool(os.getenv("OPENAI_API_KEY")), "llm_provider": provider,
                             "places_enabled": bool(os.getenv("GOOGLE_PLACES_API_KEY")),
                             "gemini_fallback_enabled": bool(os.getenv("GEMINI_API_KEY")),
                             "gemini_fallback_model": os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
                             "groq_fallback_enabled": bool(os.getenv("GROQ_API_KEY")),
                             "groq_fallback_model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                             "aviationstack_enabled": bool(os.getenv("AVIATIONSTACK_API_KEY")),
                             "irctc_key_configured": bool(os.getenv("IRCTC_API_KEY")),
                             "irctc_endpoint_configured": bool(os.getenv("IRCTC_API_BASE_URL")),
                             "railradar_enabled": bool(os.getenv("IRCTC_API_KEY") and os.getenv("IRCTC_API_BASE_URL")),
                             "transport_live": bool(os.getenv("AVIATIONSTACK_API_KEY") or
                                                    (os.getenv("IRCTC_API_KEY") and os.getenv("IRCTC_API_BASE_URL")))})
            return
        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        path = self.path.rstrip("/")
        if path not in {"/api/chat", "/api/transport/options"}:
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 65536:
                self._send_json({"error": "Request body must be between 1 byte and 64 KiB."}, HTTPStatus.BAD_REQUEST)
                return
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                self._send_json({"error": "Request body must be a JSON object."}, HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/transport/options":
                origin, destination = payload.get("origin"), payload.get("destination")
                if not isinstance(origin, str) or not isinstance(destination, str) or not origin.strip() or not destination.strip():
                    self._send_json({"error": "Origin and destination are required."}, HTTPStatus.BAD_REQUEST)
                    return
                if len(origin) > 100 or len(destination) > 100:
                    self._send_json({"error": "City names must be at most 100 characters."}, HTTPStatus.BAD_REQUEST)
                    return
                travel_date = payload.get("date")
                if travel_date is not None:
                    if not isinstance(travel_date, str) or len(travel_date) != 10:
                        self._send_json({"error": "Date must use YYYY-MM-DD format."}, HTTPStatus.BAD_REQUEST)
                        return
                    try:
                        date.fromisoformat(travel_date)
                    except ValueError:
                        self._send_json({"error": "Date must use YYYY-MM-DD format."}, HTTPStatus.BAD_REQUEST)
                        return
                self._send_json(get_transport_options(origin, destination, travel_date))
                return
            message = payload.get("message", "")
            if not isinstance(message, str) or len(message) > 4000:
                self._send_json({"error": "Message must be text of at most 4,000 characters."}, HTTPStatus.BAD_REQUEST)
                return
            message = message.strip()
            context = payload.get("context") or {}
            if not message:
                self._send_json({"error": "A message is required."}, HTTPStatus.BAD_REQUEST)
                return
            if not isinstance(context, dict):
                self._send_json({"error": "Context must be an object."}, HTTPStatus.BAD_REQUEST)
                return
            # Use the globally stored Streamlit agent/context if available;
            # fall back to a fresh agent if none exists (e.g., direct API call).
            try:
                from streamlit import runtime
                session_id = runtime.get_script_run_ctx().session_id if hasattr(runtime, "get_script_run_ctx") else None
                agent = st.session_state.get("agent", RajasthanAgent())
                ctx = st.session_state.get("context", {})
            except Exception:
                agent = RajasthanAgent()
                ctx = {}
            self._send_json(agent.run(message, ctx))
        except (ValueError, TypeError, json.JSONDecodeError):
            self._send_json({"error": "Request body must be valid JSON."}, HTTPStatus.BAD_REQUEST)
        except Exception:
            self._send_json({"error": "The planner could not complete that request. Please try again."}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format, *args):
        if os.getenv("API_REQUEST_LOGS"):
            super().log_message(format, *args)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("API_HOST", "127.0.0.1")
    print(f"Rajwada Travel Agent API listening on http://{host}:{port}")
    try:
        server = ThreadingHTTPServer((host, port), ApiHandler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            try:
                with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1) as response:
                    health = json.loads(response.read())
                if isinstance(health, dict) and health.get("status") == "ok":
                    print(f"Planner API is already running on port {port}; keeping the existing server.")
                    raise SystemExit(0)
            except (URLError, TimeoutError, ValueError, TypeError):
                pass
            print(f"Cannot start the planner API: port {port} is already in use by another process.")
        else:
            print(f"Cannot start the planner API on {host}:{port}: {exc}")
        raise SystemExit(1)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nRajwada Travel Agent API stopped.")
    finally:
        server.server_close()
