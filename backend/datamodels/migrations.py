from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Index, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime


class RefreshToken(BaseModel):
    """
    Refresh token store for rotation and security.
    Each refresh token is stored with its user_id and expiration.
    When a token is used, it's deleted from this store (rotation).
    If an old token is reused, all tokens for that user are revoked.
    """
    id: str
    user_id: str
    token_hash: str
    expires_at: datetime
    created_at: datetime
    used_at: Optional[datetime] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    class Config:
        from_attributes = True


def init_db():
    """Initialize database tables"""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    print("Database tables created/initialized.")


def init_db_with_migrations():
    """Initialize database with migrations"""
    engine = get_engine()

    # Run migrations
    with engine.connect() as conn:
        # Create refresh_tokens table if not exists
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS refresh_tokens (
                id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                token_hash VARCHAR(255) NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                used_at TIMESTAMP,
                ip_address VARCHAR(45),
                user_agent VARCHAR(255),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """))

        # Create indexes
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_refresh_tokens_user_id ON refresh_tokens(user_id)
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_refresh_tokens_expires ON refresh_tokens(expires_at)
        """))

        conn.commit()

    print("Database initialized with migrations.")


def get_engine():
    """Create database engine from connection string"""
    connection_string = os.getenv("POSTGRES_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("POSTGRES_CONNECTION_STRING not set")
    return create_engine(connection_string, pool_size=10, max_overflow=20)
