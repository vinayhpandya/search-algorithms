"""Local development settings. Never put cloud credentials into .env."""
from __future__ import annotations
import os
from pathlib import Path


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path=Path.cwd() / '.env', override=False)


def setting(name: str, default: str | None = None) -> str:
    load_env()
    value = os.environ.get(name, default)
    if not value:
        raise ValueError(f"Missing {name}; set it in .env or pass the CLI argument")
    return value
