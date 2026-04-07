from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timedelta
from datamodels.database import RefreshToken, User
from datamodels.migrations import get_engine
from sqlalchemy.orm import Session
from sqlalchemy import text
import hashlib
import secrets
from jose import jwt
from passlib.context import CryptContext
import os

# JWT Configuration
# In production, use environment variable
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15  # Updated from 30 to 15

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_token(token: str) -> str:
    """Hash a token for secure storage"""
    return hashlib.sha256(token.encode()).hexdigest()


def create_token_pair(user_id: str) -> tuple[str, str]:
    """Create a new access token and refresh token pair"""
    access_token = jwt.encode(
        {"sub": user_id, "exp": datetime.utcnow(
        ) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)},
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    refresh_token = secrets.token_urlsafe(256)
    return access_token, refresh_token


def validate_access_token(token: str) -> Optional[str]:
    """Validate an access token and return user_id if valid"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except jwt.JWTError:
        return None


def validate_refresh_token(token: str, user_id: Optional[str] = None) -> Optional[RefreshToken]:
    """
    Validate a refresh token and return the RefreshToken object if valid.
    Returns None if token is invalid, expired, or not found.
    """
    token_hash = hash_token(token)

    # Query database for refresh token
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT rt.* FROM refresh_tokens rt
            WHERE rt.token_hash = :token_hash
            AND rt.expires_at > NOW()
            AND (rt.used_at IS NULL OR rt.used_at = NOW())
        """, {"token_hash": token_hash}).fetchone())

        if result:
            return RefreshToken(**result.dict())
        return None


def revoke_refresh_token(token: str) -> bool:
    """Revoke a refresh token by hash"""
    token_hash = hash_token(token)
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            DELETE FROM refresh_tokens
            WHERE token_hash = :token_hash
        """, {"token_hash": token_hash}))
        conn.commit()
        return True


def revoke_user_refresh_tokens(user_id: str) -> int:
    """Revoke all refresh tokens for a user (security breach protocol)"""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("""
            DELETE FROM refresh_tokens
            WHERE user_id = :user_id
        """, {"user_id": user_id}))
        conn.commit()
        return result.rowcount


def create_refresh_token(user_id: str, ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> RefreshToken:
    """Create and store a new refresh token"""
    refresh_token = secrets.token_urlsafe(256)
    token_hash = hash_token(refresh_token)
    expires_at = datetime.utcnow() + timedelta(days=7)

    token = RefreshToken(
        id=secrets.token_urlsafe(32),
        user_id=user_id,
        token_hash=token_hash,
        expires_at=expires_at,
        created_at=datetime.utcnow(),
        ip_address=ip_address,
        user_agent=user_agent
    )

    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO refresh_tokens (id, user_id, token_hash, expires_at, created_at, ip_address, user_agent)
            VALUES (:id, :user_id, :token_hash, :expires_at, :created_at, :ip_address, :user_agent)
        """, {
            "id": token.id,
            "user_id": token.user_id,
            "token_hash": token.token_hash,
            "expires_at": token.expires_at,
            "created_at": token.created_at,
            "ip_address": token.ip_address,
            "user_agent": token.user_agent
        }))
        conn.commit()
        return token


def get_refresh_token_by_id(token_id: str) -> Optional[RefreshToken]:
    """Get a refresh token by its ID"""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT * FROM refresh_tokens
            WHERE id = :token_id
        """, {"token_id": token_id}).fetchone())

        if result:
            return RefreshToken(**result.dict())
        return None
