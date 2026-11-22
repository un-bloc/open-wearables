"""Tests for EncryptedString (provider tokens encrypted at rest) and the TOKEN_ENCRYPTION_KEY setting."""

import os
import subprocess
import sys
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet, InvalidToken
from pydantic import SecretStr, ValidationError
from sqlalchemy import text
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from app.config import Settings, settings
from app.models import UserConnection
from app.repositories.user_connection_repository import UserConnectionRepository
from app.utils.encrypted_string import ENCRYPTED_PREFIX, EncryptedString
from tests.factories import UserConnectionFactory

BACKEND_DIR = Path(__file__).resolve().parents[2]
TOKEN = "oura-access-token-123"


def _new_key() -> str:
    return Fernet.generate_key().decode()


def _use_keys(*keys: str) -> AbstractContextManager[Any]:
    """Run with these TOKEN_ENCRYPTION_KEY entries, the first one encrypts."""
    return patch.object(settings, "token_encryption_key", SecretStr(",".join(keys)))


def _write(value: str | None) -> str | None:
    """What the column stores for ``value``."""
    return EncryptedString().process_bind_param(value, postgresql.dialect())


def _read(stored: str | None) -> str | None:
    """What the model gets for a stored value."""
    return EncryptedString().process_result_value(stored, postgresql.dialect())


def _stored_tokens(db: Session, connection: UserConnection) -> tuple[str | None, str | None]:
    row = db.execute(
        text("SELECT access_token, refresh_token FROM user_connection WHERE id = :id"), {"id": connection.id}
    ).one()
    return row.access_token, row.refresh_token


class TestEncryptedString:
    def test_round_trip(self) -> None:
        # Act
        stored = _write(TOKEN)

        # Assert
        assert stored is not None
        assert stored.startswith(ENCRYPTED_PREFIX)
        assert TOKEN not in stored
        assert _read(stored) == TOKEN

    def test_none_stays_none(self) -> None:
        assert _write(None) is None
        assert _read(None) is None

    @pytest.mark.parametrize("token", ["", "é" * 10 + "x" * 5000, "enc1:looks-encrypted"])
    def test_any_string_round_trips(self, token: str) -> None:
        assert _read(_write(token)) == token

    def test_the_same_token_is_stored_differently_each_time(self) -> None:
        """Why the column can't be compared or filtered in SQL."""
        assert _write(TOKEN) != _write(TOKEN)

    def test_legacy_plain_text_is_read_as_it_is(self) -> None:
        assert _read(TOKEN) == TOKEN

    def test_legacy_plain_text_is_encrypted_when_written_again(self) -> None:
        stored = _write(_read(TOKEN))

        assert stored is not None
        assert stored.startswith(ENCRYPTED_PREFIX)
        assert _read(stored) == TOKEN

    def test_data_written_with_the_old_key_reads_after_a_new_key_is_added_in_front(self) -> None:
        # Arrange
        old, new = _new_key(), _new_key()
        with _use_keys(old):
            written_with_old = _write(TOKEN)

        # Act
        with _use_keys(new, old):
            read_back = _read(written_with_old)
            written_with_new = _write(TOKEN)

        # Assert
        assert read_back == TOKEN
        assert written_with_new is not None
        ciphertext = written_with_new.removeprefix(ENCRYPTED_PREFIX)
        assert Fernet(new).decrypt(ciphertext) == TOKEN.encode()
        with pytest.raises(InvalidToken):
            Fernet(old).decrypt(ciphertext)

    def test_dropping_the_old_key_while_data_still_uses_it_fails_loudly(self) -> None:
        old, new = _new_key(), _new_key()
        with _use_keys(old):
            written_with_old = _write(TOKEN)

        with _use_keys(new), pytest.raises(RuntimeError, match="TOKEN_ENCRYPTION_KEY"):
            _read(written_with_old)

    def test_wrong_key_fails_loudly(self) -> None:
        with _use_keys(_new_key()):
            stored = _write(TOKEN)

        with _use_keys(_new_key()), pytest.raises(RuntimeError, match="cannot be decrypted") as excinfo:
            _read(stored)

        assert TOKEN not in str(excinfo.value)

    @pytest.mark.parametrize(
        "stored", [f"{ENCRYPTED_PREFIX}not-a-fernet-token", f"{ENCRYPTED_PREFIX}é", ENCRYPTED_PREFIX]
    )
    def test_a_corrupted_value_fails_loudly_instead_of_being_taken_for_a_token(self, stored: str) -> None:
        with pytest.raises(RuntimeError, match="cannot be decrypted"):
            _read(stored)

    def test_without_a_key_nothing_is_stored_as_plain_text(self) -> None:
        with _use_keys(""), pytest.raises(RuntimeError, match="TOKEN_ENCRYPTION_KEY is not set"):
            _write(TOKEN)


class TestUserConnectionTokens:
    """UserConnection tokens through the ORM, the repository and raw SQL."""

    @pytest.fixture
    def repo(self) -> UserConnectionRepository:
        return UserConnectionRepository(UserConnection)

    def test_tokens_are_encrypted_in_the_row_and_plain_on_the_model(self, db: Session) -> None:
        # Arrange / Act
        connection = UserConnectionFactory(access_token="access-1", refresh_token="refresh-1")
        stored_access, stored_refresh = _stored_tokens(db, connection)
        db.expire(connection)

        # Assert
        assert stored_access is not None
        assert stored_refresh is not None
        assert stored_access.startswith(ENCRYPTED_PREFIX)
        assert stored_refresh.startswith(ENCRYPTED_PREFIX)
        assert "access-1" not in stored_access
        assert "refresh-1" not in stored_refresh
        assert connection.access_token == "access-1"
        assert connection.refresh_token == "refresh-1"

    def test_a_connection_without_tokens_stores_null(self, db: Session, repo: UserConnectionRepository) -> None:
        user_id = UserConnectionFactory().user_id

        connection, _ = repo.ensure_sdk_connection(db, user_id, "apple")

        assert _stored_tokens(db, connection) == (None, None)
        assert connection.access_token is None

    def test_a_legacy_plain_text_row_reads_and_is_encrypted_by_the_next_write(
        self, db: Session, repo: UserConnectionRepository
    ) -> None:
        # Arrange: a row as 0.11.0 wrote it
        connection = UserConnectionFactory()
        db.execute(
            text("UPDATE user_connection SET access_token = :access, refresh_token = :refresh WHERE id = :id"),
            {"access": "legacy-access", "refresh": "legacy-refresh", "id": connection.id},
        )
        db.expire(connection)
        assert connection.access_token == "legacy-access"
        assert connection.refresh_token == "legacy-refresh"

        # Act: a token refresh
        repo.update_tokens(db, connection, "new-access", "new-refresh", expires_in=3600)

        # Assert
        stored_access, stored_refresh = _stored_tokens(db, connection)
        assert stored_access is not None
        assert stored_refresh is not None
        assert stored_access.startswith(ENCRYPTED_PREFIX)
        assert stored_refresh.startswith(ENCRYPTED_PREFIX)
        db.expire(connection)
        assert (connection.access_token, connection.refresh_token) == ("new-access", "new-refresh")

    def test_a_token_another_worker_wrote_is_read_by_reload_tokens(
        self, db: Session, repo: UserConnectionRepository
    ) -> None:
        connection = UserConnectionFactory(access_token="stale")
        # Raw SQL, like another worker's session: the instance in this session stays stale
        db.execute(
            text("UPDATE user_connection SET access_token = :stored WHERE id = :id"),
            {"stored": _write("refreshed-elsewhere"), "id": connection.id},
        )

        repo.reload_tokens(db, connection)

        assert connection.access_token == "refreshed-elsewhere"

    def test_updating_another_column_does_not_rewrite_the_tokens(self, db: Session) -> None:
        connection = UserConnectionFactory()
        before = _stored_tokens(db, connection)

        connection.last_synced_at = datetime.now(timezone.utc)
        db.flush()

        assert _stored_tokens(db, connection) == before

    def test_disconnect_clears_both_tokens(self, db: Session, repo: UserConnectionRepository) -> None:
        connection = UserConnectionFactory()

        repo.disconnect(db, connection.user_id, connection.provider)

        assert _stored_tokens(db, connection) == (None, None)

    def test_a_stored_value_that_does_not_decrypt_fails_the_read(self, db: Session) -> None:
        connection = UserConnectionFactory()
        db.execute(
            text("UPDATE user_connection SET access_token = :stored WHERE id = :id"),
            {"stored": f"{ENCRYPTED_PREFIX}garbage", "id": connection.id},
        )
        db.expire(connection)

        with pytest.raises(RuntimeError, match="cannot be decrypted"):
            _ = connection.access_token


class TestTokenEncryptionKeySetting:
    def test_a_missing_key_fails_with_the_variable_name_and_how_to_generate_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TOKEN_ENCRYPTION_KEY")

        with pytest.raises(RuntimeError, match="TOKEN_ENCRYPTION_KEY is not set") as excinfo:
            Settings(_env_file=None)

        assert "Fernet.generate_key()" in str(excinfo.value)

    @pytest.mark.parametrize("value", ["", "   "])
    def test_a_blank_key_counts_as_missing(self, value: str) -> None:
        with pytest.raises(RuntimeError, match="TOKEN_ENCRYPTION_KEY is not set"):
            Settings(token_encryption_key=value)

    def test_a_malformed_key_is_rejected_without_echoing_it(self) -> None:
        with pytest.raises(RuntimeError, match="entry 1 is not a Fernet key") as excinfo:
            Settings(token_encryption_key="not-a-fernet-key")

        assert "not-a-fernet-key" not in str(excinfo.value)

    def test_a_key_one_character_short_is_rejected_without_echoing_it(self) -> None:
        key = _new_key()[:-1]

        with pytest.raises(RuntimeError, match="entry 1") as excinfo:
            Settings(token_encryption_key=key)

        assert key not in str(excinfo.value)

    @pytest.mark.parametrize("suffix", [",oops", ",", ",,"])
    def test_the_bad_entry_of_a_list_is_named(self, suffix: str) -> None:
        with pytest.raises(RuntimeError, match="entry 2 is not a Fernet key"):
            Settings(token_encryption_key=_new_key() + suffix)

    def test_one_key_or_a_list_is_accepted_and_whitespace_around_entries_is_ignored(self) -> None:
        first, second = _new_key(), _new_key()

        assert Settings(token_encryption_key=first).token_encryption_key.get_secret_value() == first
        Settings(token_encryption_key=f" {first} , {second}\n")

    def test_an_error_about_another_setting_does_not_print_the_input_it_was_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # For a missing field pydantic prints the whole input, every value in it, unless told not to.
        monkeypatch.delenv("SECRET_KEY")
        key = _new_key()

        with pytest.raises(ValidationError) as excinfo:
            Settings(_env_file=None, token_encryption_key=key)

        message = str(excinfo.value)
        assert "secret_key" in message
        assert "input_value" not in message
        assert key[-12:] not in message

    def test_the_key_is_not_shown_in_the_settings_repr(self) -> None:
        key = _new_key()

        assert key not in repr(Settings(token_encryption_key=key))


class TestStartup:
    """The app is not started without a usable key, whichever process imports it (API, worker, beat, alembic)."""

    @staticmethod
    def _import_app(**env: str) -> subprocess.CompletedProcess[str]:
        child_env = {name: value for name, value in os.environ.items() if name != "TOKEN_ENCRYPTION_KEY"} | env
        return subprocess.run(
            [sys.executable, "-c", "import app.main"],
            cwd=BACKEND_DIR,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def test_the_app_does_not_start_without_a_key(self) -> None:
        result = self._import_app()

        assert result.returncode != 0
        assert "TOKEN_ENCRYPTION_KEY is not set" in result.stderr

    def test_the_app_does_not_start_with_a_malformed_key_and_does_not_log_it(self) -> None:
        result = self._import_app(TOKEN_ENCRYPTION_KEY="not-a-fernet-key")

        assert result.returncode != 0
        assert "TOKEN_ENCRYPTION_KEY entry 1 is not a Fernet key" in result.stderr
        assert "not-a-fernet-key" not in result.stderr

    def test_the_app_starts_with_a_key(self) -> None:
        result = self._import_app(TOKEN_ENCRYPTION_KEY=_new_key())

        assert result.returncode == 0, result.stderr
