"""
langchain_browserlm/core/config.py
======================
Application settings loaded from .env.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr


# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

APP_DIR = Path(__file__).resolve().parents[2]

ENV_PATH = APP_DIR / ".env"
PROFILES_DIR = APP_DIR / ".profiles"
QWEN_URL = "https://chat.qwen.ai"

DEFAULT_MODEL = "Qwen3.6-Plus"
DEFAULT_HEADLESS = "0"
DEFAULT_MODEL_MODE = "Auto"


# â”€â”€ Models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ModelMode(StrEnum):
    THINKING = "Thinking"
    AUTO = "Auto"
    FAST = "Fast"


_VALID_REGIMES = {mode.value for mode in ModelMode}


class UserCred(BaseModel):
    index: int = Field(ge=1)
    login: str
    password: SecretStr


class Settings(BaseModel):
    model: str
    is_headless: bool
    model_mode: ModelMode
    creds: list[UserCred]


# â”€â”€ Raw .env loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    env: dict[str, str] = {}

    if not path.exists():
        return env

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")

    return env


# â”€â”€ Parsing helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def parse_credentials(env: dict[str, str]) -> list[UserCred]:
    creds: list[UserCred] = []
    index = 1

    while True:
        login = env.get(f"USER{index}_LOGIN")
        password = env.get(f"USER{index}_PASSWORD")

        if not login or not password:
            break

        creds.append(UserCred(index=index, login=login, password=password))
        index += 1

    return creds


def parse_model_mode(value: str) -> ModelMode:
    try:
        return ModelMode(value)
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in ModelMode)
        raise ValueError(
            f"Invalid MODEL_MODE: {value!r}. Expected one of: {allowed}"
        ) from exc


def parse_headless(value: str) -> bool:
    v = value.lower()
    if v in {"1", "true", "yes"}:
        return True
    if v in {"0", "false", "no"}:
        return False
    raise ValueError(f"Invalid HEADLESS value: {value!r}. Expected 0/1 or true/false.")


def get_model(env: dict[str, str]) -> str:
    return env.get("MODEL", DEFAULT_MODEL).strip()


def get_model_regime(env: dict[str, str]) -> str:
    return parse_model_mode(
        env.get("MODEL_MODE", DEFAULT_MODEL_MODE).strip()
    ).value


def get_headless(env: dict[str, str]) -> bool:
    return parse_headless(env.get("HEADLESS", DEFAULT_HEADLESS).strip())


# â”€â”€ Public settings loader â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_settings(path: Path = ENV_PATH) -> Settings:
    env = load_env(path)

    return Settings(
        model=env.get("MODEL", DEFAULT_MODEL).strip(),
        is_headless=parse_headless(env.get("HEADLESS", DEFAULT_HEADLESS).strip()),
        model_mode=parse_model_mode(env.get("MODEL_MODE", DEFAULT_MODEL_MODE).strip()),
        creds=parse_credentials(env),
    )
