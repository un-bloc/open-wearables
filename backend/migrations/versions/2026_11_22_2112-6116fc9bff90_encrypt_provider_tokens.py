"""encrypt provider tokens

Encrypts user_connection.access_token and refresh_token in place, in the format of app/utils/encrypted_string.py:
"enc1:" + a Fernet token, the keys from TOKEN_ENCRYPTION_KEY. A token that already carries the prefix is left
alone, so running it again changes nothing. The downgrade decrypts them: a release without this column type
would send the ciphertext to the provider and revoke the connection.

Both directions need a database connection (no --sql) and the TOKEN_ENCRYPTION_KEY the tokens are encrypted with.

Revision ID: 6116fc9bff90
Revises: e9413ac05e3b

"""

import logging
from typing import Callable, Sequence, Union

import sqlalchemy as sa
from alembic import op
from cryptography.fernet import Fernet, InvalidToken, MultiFernet

# revision identifiers, used by Alembic.
revision: str = "6116fc9bff90"
down_revision: Union[str, None] = "e9413ac05e3b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Frozen copy of the stored format (app/utils/encrypted_string.py), so later changes to the app can't alter this
PREFIX = "enc1:"

# Rows with a token still to convert
_PLAIN = (
    "(access_token IS NOT NULL AND access_token NOT LIKE :prefix)"
    " OR (refresh_token IS NOT NULL AND refresh_token NOT LIKE :prefix)"
)
_ENCRYPTED = "access_token LIKE :prefix OR refresh_token LIKE :prefix"

log = logging.getLogger("alembic.runtime.migration")


def upgrade() -> None:
    encrypt_tokens(_connection(), _cipher())


def downgrade() -> None:
    decrypt_tokens(_connection(), _cipher())


def encrypt_tokens(connection: sa.Connection, cipher: MultiFernet) -> None:
    _convert(connection, _PLAIN, lambda value: _encrypt(cipher, value))


def decrypt_tokens(connection: sa.Connection, cipher: MultiFernet) -> None:
    _convert(connection, _ENCRYPTED, lambda value: _decrypt(cipher, value))


def _connection() -> sa.Connection:
    if op.get_context().as_sql:
        raise RuntimeError("Converting the stored tokens needs a database connection: run this migration online.")
    return op.get_bind()


def _cipher() -> MultiFernet:
    # Imported here, so that loading the migration chain (alembic heads, scripts/check_migrations.py) needs no settings
    from app.config import settings

    return MultiFernet([Fernet(key.strip()) for key in settings.token_encryption_key.get_secret_value().split(",")])


def _encrypt(cipher: MultiFernet, value: str | None) -> str | None:
    if value is None or value.startswith(PREFIX):
        return value
    return PREFIX + cipher.encrypt(value.encode()).decode()


def _decrypt(cipher: MultiFernet, value: str | None) -> str | None:
    if value is None or not value.startswith(PREFIX):
        return value
    try:
        return cipher.decrypt(value.removeprefix(PREFIX).encode()).decode()
    except (InvalidToken, ValueError):
        raise RuntimeError(
            "A stored token cannot be decrypted with the keys in TOKEN_ENCRYPTION_KEY. "
            "Run this migration with the key list the tokens were written with."
        ) from None


def _convert(connection: sa.Connection, where: str, convert: Callable[[str | None], str | None]) -> None:
    # FOR UPDATE: a token refresh running meanwhile waits for this transaction, not overwritten by a stale value
    rows = connection.execute(
        sa.text(f"SELECT id, access_token, refresh_token FROM user_connection WHERE {where} FOR UPDATE"),
        {"prefix": PREFIX + "%"},
    ).all()
    for row_id, access_token, refresh_token in rows:
        connection.execute(
            sa.text("UPDATE user_connection SET access_token = :access, refresh_token = :refresh WHERE id = :id"),
            {"id": row_id, "access": convert(access_token), "refresh": convert(refresh_token)},
        )
    log.info("Converted the tokens of %d connection(s)", len(rows))
