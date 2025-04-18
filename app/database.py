from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from contextlib import contextmanager

# Get the SQLite database URL from environment variables or use a default
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./emails.db")

# Create SQLAlchemy engine with check_same_thread=False for SQLite
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create a SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a Base class for declarative models
Base = declarative_base()

@contextmanager
def get_db():
    """Dependency function to get a DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize the database by creating all tables"""
    from app.models.database import Base
    Base.metadata.create_all(bind=engine)
    print("Database initialized with all tables created")