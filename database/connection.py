import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Generator

# Fetch Database URL from environment variables, fallback to local SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/churn_intelligence.db")

# Ensure the local data directory exists if using default SQLite path
if DATABASE_URL.startswith("sqlite:///./data/"):
    os.makedirs("./data", exist_ok=True)

# Configuration for SQLite to allow multi-threading in FastAPI
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

# Create SQLAlchemy engine and session factory
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for database models
Base = declarative_base()


def init_db() -> None:
    """
    Utility function to create all database tables.
    Should be called during application startup.
    """
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator:
    """
    Dependency generator function to yield a database session.
    Automatically closes the session after the FastAPI request lifecycle.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
