from __future__ import annotations

import logging
import os
from typing import Iterable


SECRET_ENV_NAMES = (
    "POLYMARKET_PRIVATE_KEY",
    "POLYMARKET_API_KEY",
    "POLYMARKET_API_SECRET",
    "POLYMARKET_API_PASSPHRASE",
    "POLYMARKET_FUNDER",
)


class SecretFilter(logging.Filter):
    """Redacts configured secrets if an exception or debug payload includes them."""

    def __init__(self, secrets: Iterable[str] | None = None) -> None:
        super().__init__()
        self.secrets = [value for value in (secrets or []) if value]

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for secret in self.secrets:
            if secret and secret in message:
                message = message.replace(secret, "***REDACTED***")
        record.msg = message
        record.args = ()
        return True


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("polymarket_btc_5m_bot")
    if logger.handlers:
        return logger

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(SecretFilter(os.getenv(name) for name in SECRET_ENV_NAMES))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
