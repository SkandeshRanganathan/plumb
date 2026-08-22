import os
import json
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Build PostgreSQL connection URL
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "cric_ai")

# Fallback to SQLite if we're not running in Docker or haven't set up Postgres locally yet
if POSTGRES_HOST == "localhost" and not os.getenv("FORCE_POSTGRES"):
    DATABASE_URL = "sqlite:///./cric_ai.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Models ---

class BowlerState(Base):
    __tablename__ = "bowler_states"
    
    id = Column(Integer, primary_key=True, index=True)
    bowler_name = Column(String, unique=True, index=True)
    last_over = Column(String)
    predicted_type = Column(String)

class OverHistory(Base):
    __tablename__ = "over_history"
    
    id = Column(Integer, primary_key=True, index=True)
    over_prefix = Column(String, unique=True, index=True)
    balls_json = Column(Text) # Store list of ball coordinates as JSON string

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
