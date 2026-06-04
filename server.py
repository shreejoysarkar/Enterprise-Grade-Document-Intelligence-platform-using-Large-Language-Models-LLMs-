"""
Server entry point — runs the FastAPI app with Uvicorn.

Usage:
    python server.py
    python server.py --host 0.0.0.0 --port 8000 --reload
"""

import argparse
import uvicorn

from utils.config import get_settings

settings = get_settings()


def main():
    parser = argparse.ArgumentParser(description="Doc-Intel RAG Platform — API Server")
    parser.add_argument("--host", default=settings.api_host, help="Bind host")
    parser.add_argument("--port", type=int, default=settings.api_port, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    print(f"\n  Doc-Intel Platform  v{settings.app_version}")
    print(f"  -> http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}\n")

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
