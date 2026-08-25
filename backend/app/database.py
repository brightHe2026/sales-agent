import os

from sqlalchemy import create_engine
from collections.abc import Generator

from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://sales@localhost:5432/sales_agent",
)


class Base(DeclarativeBase):
    pass


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
