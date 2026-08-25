import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app import models  # noqa: F401


@pytest.fixture()
def session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    event.listen(engine, "connect", lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"))
    Base.metadata.create_all(engine)
    local_session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        yield local_session
    finally:
        local_session.close()
        engine.dispose()
