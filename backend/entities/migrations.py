from sqlalchemy.engine import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import QueuePool
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from shared.settings import get_settings


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


async def init_db():
    """Initialize database tables"""
    async_engine = get_async_engine()
    async with async_engine.begin() as conn:
        await conn.run_sync(DeclarativeBase.metadata.create_all)
    print("Database tables created/initialized.")


async def init_db_with_migrations():
    """Initialize database with migrations"""
    async_engine = get_async_engine()

    # Run migrations
    async with async_engine.begin() as conn:
        # Create refresh_tokens table if not exists
        await conn.run_sync(text("""
            CREATE TABLE IF NOT EXISTS refresh_tokens (
                id VARCHAR(255) PRIMARY KEY,
                user_id UUID NOT NULL,
                token_hash VARCHAR(255) NOT NULL,
                expires_at TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                used_at TIMESTAMPTZ,
                ip_address VARCHAR(45),
                user_agent VARCHAR(255),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """))

        # Create indexes
        await conn.run_sync(text("""
            CREATE INDEX IF NOT EXISTS ix_refresh_tokens_user_id ON refresh_tokens(user_id)
        """))

        await conn.run_sync(text("""
            CREATE INDEX IF NOT EXISTS ix_refresh_tokens_expires ON refresh_tokens(expires_at)
        """))

    print("Database initialized with migrations.")


def get_engine():
    """Create database engine from connection string (sync, for metadata)"""
    settings = get_settings()
    connection_string = settings.postgres_connection_string
    if not connection_string:
        raise ValueError("POSTGRES_CONNECTION_STRING not set")
    return create_engine(
        connection_string,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        echo=settings.database_echo
    )


def get_async_engine():
    """Create async database engine from connection string"""
    settings = get_settings()
    connection_string = settings.async_postgres_connection_string
    if not connection_string:
        raise ValueError("ASYNC_POSTGRES_CONNECTION_STRING not set")
    return create_async_engine(
        connection_string,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        echo=settings.database_echo
    )
