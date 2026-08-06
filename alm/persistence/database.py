"""Engine, session factory and schema management.

``ALM_DATABASE_URL`` selects the backend (SQLite by default, PostgreSQL in
production).  :func:`init_db` creates missing tables and adds columns that
appeared in the models since the database was created — enough auto-migration
to keep a single-node deployment moving without adopting Alembic, and no more
than that: destructive changes are never inferred.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from alm.config import get_settings
from alm.persistence.models import Base

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine(url: str | None = None) -> Engine:
    """Return (and lazily create) the process-wide engine."""
    global _engine
    if _engine is None or url is not None:
        resolved = url or get_settings().database_url
        kwargs: dict[str, object] = {"echo": False, "future": True}
        if resolved.startswith("sqlite"):
            # Allow use from the API's worker threads.
            kwargs["connect_args"] = {"check_same_thread": False}
        engine = create_engine(resolved, **kwargs)  # type: ignore[arg-type]
        if url is not None:
            return engine
        _engine = engine
        logger.info("Database engine created: %s", resolved.split("@")[-1])
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory


def get_session() -> Session:
    """Return a new SQLAlchemy session (caller owns closing it)."""
    return get_session_factory()()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commits on success, rolls back on error."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _add_missing_columns(engine: Engine) -> None:
    """Issue ``ALTER TABLE … ADD COLUMN`` for columns absent from the database."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue  # create_all will handle it
        existing = {c["name"] for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing:
                continue
            col_type = column.type.compile(dialect=engine.dialect)
            # A new column on a populated table cannot be NOT NULL without a
            # default, so widen it rather than fail the migration.
            ddl = f"ALTER TABLE {table.name} ADD COLUMN {column.name} {col_type} NULL"
            logger.info("Auto-migrating: %s", ddl)
            with engine.begin() as conn:
                conn.execute(text(ddl))


def init_db(engine: Engine | None = None) -> None:
    """Ensure the schema exists and is up to date."""
    eng = engine or get_engine()
    _add_missing_columns(eng)
    Base.metadata.create_all(eng)
    logger.debug("Database schema ensured")


def reset_engine() -> None:
    """Drop cached engine/session factory — used by tests switching databases."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None


def drop_all(engine: Engine | None = None) -> None:
    """Drop every ALM table.  Destructive; only exposed for tests and ``alm db reset``."""
    Base.metadata.drop_all(engine or get_engine())
