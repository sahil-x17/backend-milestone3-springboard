from pathlib import Path

from sqlalchemy import Column, Integer, String, create_engine, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "app.db"

SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)  # hashed password
    profile_picture = Column(String, nullable=True)  # file path on disk
    index = Column(String, nullable=True)  # Pinecone index name: firstname_lastname_id


def init_db() -> None:
    """Create all tables if they do not exist."""
    Base.metadata.create_all(bind=engine)
    # Ensure new columns exist when DB already created (simple SQLite migration)
    inspector = inspect(engine)
    if inspector.has_table("users"):
        existing_cols = {col["name"] for col in inspector.get_columns("users")}
        if "index" not in existing_cols:
            with engine.connect() as conn:
                conn.execute(text('ALTER TABLE users ADD COLUMN "index" VARCHAR'))
                conn.commit()


