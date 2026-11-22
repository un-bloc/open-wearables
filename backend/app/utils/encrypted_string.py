from cryptography.fernet import InvalidToken, MultiFernet
from sqlalchemy import Dialect, Text
from sqlalchemy.types import TypeDecorator

from app.config import build_token_cipher, settings

# Starts every encrypted value and versions the format: a stored value without it is legacy plain text.
ENCRYPTED_PREFIX = "enc1:"


def _cipher() -> MultiFernet:
    return build_token_cipher(settings.token_encryption_key.get_secret_value())


class EncryptedString(TypeDecorator[str]):
    """A secret column: the row holds ``enc1:`` + a Fernet token (keys in TOKEN_ENCRYPTION_KEY), the model the value.

    A stored value without the prefix is legacy plain text: it is read as it is and encrypted the next time
    it is written. A prefixed value that doesn't decrypt raises, it is never passed on as a token.
    Fernet output differs on every call, so don't compare or filter this column in SQL (IS NULL is fine).
    """

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: str | None, dialect: Dialect) -> str | None:
        if value is None:
            return None
        return ENCRYPTED_PREFIX + _cipher().encrypt(value.encode()).decode()

    def process_result_value(self, value: str | None, dialect: Dialect) -> str | None:
        if value is None or not value.startswith(ENCRYPTED_PREFIX):
            return value
        try:
            return _cipher().decrypt(value.removeprefix(ENCRYPTED_PREFIX).encode()).decode()
        except (InvalidToken, ValueError):
            raise RuntimeError(
                "A stored token cannot be decrypted with the keys in TOKEN_ENCRYPTION_KEY. "
                "Keep every key that tokens were written with in that list, the newest first."
            ) from None
