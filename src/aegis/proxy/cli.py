from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AegisLLM inline security gateway.")
    parser.add_argument("--host", default=os.environ.get("AEGIS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AEGIS_PORT", "8000")))
    parser.add_argument("--log-level", default=os.environ.get("AEGIS_LOG_LEVEL", "info"))
    args = parser.parse_args()

    uvicorn.run(
        "aegis.proxy.app:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        workers=1,
    )


if __name__ == "__main__":
    main()
