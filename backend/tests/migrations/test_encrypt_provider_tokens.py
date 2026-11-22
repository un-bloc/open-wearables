"""Tests for the migration that encrypts the stored provider tokens (and decrypts them on downgrade)."""

import importlib.util
import io
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from cryptography.fernet import Fernet, MultiFernet
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import build_token_cipher, settings
from app.models import UserConnection
from app.utils.encrypted_string import ENCRYPTED_PREFIX
from tests.factories import UserConnectionFactory

_MIGRATION_PATH = next((Path(__file__).resolve().parents[2] / "migrations" / "versions").glob("*-6116fc9bff90_*.py"))


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("encrypt_provider_tokens_migration", _MIGRATION_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


migration = _load_migration()


@pytest.fixture
def cipher() -> MultiFernet:
    return build_token_cipher(settings.token_encryption_key.get_secret_value())


def _connection_with_raw_tokens(db: Session, access: str | None, refresh: str | None) -> UserConnection:
    """A connection whose tokens are stored exactly as given, as 0.11.0 stored them (plain text)."""
    connection = UserConnectionFactory()
    db.execute(
        text("UPDATE user_connection SET access_token = :access, refresh_token = :refresh WHERE id = :id"),
        {"access": access, "refresh": refresh, "id": connection.id},
    )
    db.expire(connection)
    return connection


def _raw_tokens(db: Session, connection: UserConnection) -> tuple[str | None, str | None]:
    row = db.execute(
        text("SELECT access_token, refresh_token FROM user_connection WHERE id = :id"), {"id": connection.id}
    ).one()
    return row.access_token, row.refresh_token


def _run_through_alembic(db: Session, step: Callable[[], None]) -> None:
    """Run upgrade() or downgrade() the way `alembic upgrade` does, on the test transaction."""
    with Operations.context(MigrationContext.configure(db.connection())):
        step()


class TestEncryptProviderTokens:
    def test_it_writes_the_format_the_column_type_reads(self) -> None:
        assert migration.PREFIX == ENCRYPTED_PREFIX

    def test_it_encrypts_plain_text_tokens(self, db: Session, cipher: MultiFernet) -> None:
        # Arrange
        legacy = _connection_with_raw_tokens(db, "legacy-access", "legacy-refresh")

        # Act
        migration.encrypt_tokens(db.connection(), cipher)

        # Assert
        stored_access, stored_refresh = _raw_tokens(db, legacy)
        assert stored_access is not None
        assert stored_refresh is not None
        assert stored_access.startswith(ENCRYPTED_PREFIX)
        assert stored_refresh.startswith(ENCRYPTED_PREFIX)
        assert "legacy-access" not in stored_access
        assert "legacy-refresh" not in stored_refresh
        db.expire(legacy)
        assert (legacy.access_token, legacy.refresh_token) == ("legacy-access", "legacy-refresh")

    def test_it_is_idempotent(self, db: Session, cipher: MultiFernet) -> None:
        legacy = _connection_with_raw_tokens(db, "legacy-access", "legacy-refresh")
        migration.encrypt_tokens(db.connection(), cipher)
        after_first_run = _raw_tokens(db, legacy)

        migration.encrypt_tokens(db.connection(), cipher)

        # Fernet output differs on every call, so an equal row was not encrypted a second time
        assert _raw_tokens(db, legacy) == after_first_run

    def test_it_leaves_null_and_already_encrypted_tokens_alone(self, db: Session, cipher: MultiFernet) -> None:
        # Arrange
        written_by_the_app = UserConnectionFactory(access_token="app-access", refresh_token="app-refresh")
        sdk_connection = _connection_with_raw_tokens(db, None, None)
        half_done = _connection_with_raw_tokens(db, "plain-access", None)
        app_before = _raw_tokens(db, written_by_the_app)

        # Act
        migration.encrypt_tokens(db.connection(), cipher)

        # Assert
        assert _raw_tokens(db, written_by_the_app) == app_before
        assert _raw_tokens(db, sdk_connection) == (None, None)
        half_access, half_refresh = _raw_tokens(db, half_done)
        assert half_access is not None
        assert half_access.startswith(ENCRYPTED_PREFIX)
        assert half_refresh is None

    def test_it_encrypts_only_the_plain_token_of_a_half_encrypted_row(self, db: Session, cipher: MultiFernet) -> None:
        # Arrange: the access token was already rewritten by the app, the refresh token not yet
        connection = UserConnectionFactory(access_token="app-access")
        encrypted_access, _ = _raw_tokens(db, connection)
        db.execute(
            text("UPDATE user_connection SET refresh_token = 'plain-refresh' WHERE id = :id"), {"id": connection.id}
        )

        # Act
        migration.encrypt_tokens(db.connection(), cipher)

        # Assert
        stored_access, stored_refresh = _raw_tokens(db, connection)
        assert stored_access == encrypted_access
        assert stored_refresh is not None
        assert stored_refresh.startswith(ENCRYPTED_PREFIX)
        db.expire(connection)
        assert (connection.access_token, connection.refresh_token) == ("app-access", "plain-refresh")

    def test_it_is_safe_on_an_empty_table(self, db: Session, cipher: MultiFernet) -> None:
        db.execute(text("DELETE FROM user_connection"))

        migration.encrypt_tokens(db.connection(), cipher)
        migration.decrypt_tokens(db.connection(), cipher)

        assert db.execute(text("SELECT count(*) FROM user_connection")).scalar() == 0

    def test_upgrade_then_downgrade_round_trips_through_alembic(self, db: Session) -> None:
        # Arrange
        legacy = _connection_with_raw_tokens(db, "legacy-access", "legacy-refresh")

        # Act / Assert
        _run_through_alembic(db, migration.upgrade)
        stored_access, stored_refresh = _raw_tokens(db, legacy)
        assert stored_access is not None
        assert stored_refresh is not None
        assert stored_access.startswith(ENCRYPTED_PREFIX)
        assert stored_refresh.startswith(ENCRYPTED_PREFIX)

        _run_through_alembic(db, migration.downgrade)
        assert _raw_tokens(db, legacy) == ("legacy-access", "legacy-refresh")

    def test_downgrade_with_another_key_fails_and_changes_nothing(self, db: Session, cipher: MultiFernet) -> None:
        # Arrange
        legacy = _connection_with_raw_tokens(db, "legacy-access", "legacy-refresh")
        migration.encrypt_tokens(db.connection(), cipher)
        encrypted = _raw_tokens(db, legacy)

        # Act / Assert
        with pytest.raises(RuntimeError, match="TOKEN_ENCRYPTION_KEY"):
            migration.decrypt_tokens(db.connection(), MultiFernet([Fernet(Fernet.generate_key())]))
        assert _raw_tokens(db, legacy) == encrypted

    def test_it_refuses_to_run_offline(self) -> None:
        offline = MigrationContext.configure(
            dialect_name="postgresql", opts={"as_sql": True, "output_buffer": io.StringIO()}
        )

        with Operations.context(offline), pytest.raises(RuntimeError, match="needs a database connection"):
            migration.upgrade()
