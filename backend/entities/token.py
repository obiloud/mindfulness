from typing import Optional
from datetime import datetime, timedelta, timezone
from entities.database import RefreshToken, User
from entities.migrations import get_async_engine
from sqlalchemy import text
import hashlib
import secrets
from jose import jwt
from passlib.context import CryptContext
import os
from fastapi import Request, HTTPException
import logging

# Configure logger
logger = logging.getLogger(__name__)

# JWT Configuration
# In production, use environment variable
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15  # Updated from 30 to 15

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def hash_token(token: str) -> str:
    """Hash a token for secure storage"""
    return hashlib.sha256(token.encode()).hexdigest()


async def create_token_pair(user_id: str, ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> tuple[str, str]:
    """Create a new access token and refresh token pair, storing refresh token in DB"""
    access_token = jwt.encode(
        {"sub": user_id, "exp": datetime.now(timezone.utc
                                             ) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)},
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    _, refresh_token = await create_refresh_token(user_id, ip_address, user_agent)
    return access_token, refresh_token


async def validate_access_token(token: str) -> Optional[str]:
    """Validate an access token and return user_id if valid"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except jwt.JWTError:
        return None


async def validate_refresh_token(token: str, request: Optional[Request] = None) -> Optional[RefreshToken]:
    """
    Validate a refresh token and return the RefreshToken object if valid.
    Returns None if token is invalid, expired, or not found.
    """
    token_hash = await hash_token(token)

    # Query database for refresh token
    async_engine = get_async_engine()
    async with async_engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT rt.* FROM refresh_tokens rt
            WHERE rt.token_hash = :token_hash
            AND rt.expires_at > NOW()
            AND (rt.used_at IS NULL OR rt.used_at = NOW())
        """), {"token_hash": token_hash})
        row = await conn.fetchfirst(result)

        if row:
            # Validate IP address and user agent if request is provided
            if request:
                current_ip = request.client.host if request.client else None
                current_user_agent = request.headers.get("user-agent")

                # Check if IP address or user agent has changed (potential token theft)
                if row.ip_address and current_ip and row.ip_address != current_ip:
                    logger.warning(
                        f"IP address mismatch for token {row.id}: "
                        f"expected {row.ip_address}, got {current_ip}"
                    )
                    # Revoke all tokens for this user (security breach protocol)
                    await revoke_user_refresh_tokens(row.user_id)
                    raise HTTPException(
                        status_code=401,
                        detail="Security breach: IP address mismatch detected"
                    )

                if row.user_agent and current_user_agent and row.user_agent != current_user_agent:
                    logger.warning(
                        f"User agent mismatch for token {row.id}: "
                        f"expected {row.user_agent}, got {current_user_agent}"
                    )
                    # Revoke all tokens for this user (security breach protocol)
                    await revoke_user_refresh_tokens(row.user_id)
                    raise HTTPException(
                        status_code=401,
                        detail="Security breach: User agent mismatch detected"
                    )

            return RefreshToken(**row.dict())
        return None


async def revoke_refresh_token(token: str) -> bool:
    """Revoke a refresh token by hash"""
    token_hash = await hash_token(token)
    async_engine = get_async_engine()
    async with async_engine.begin() as conn:
        await conn.execute(text("""
            DELETE FROM refresh_tokens
            WHERE token_hash = :token_hash
        """), {"token_hash": token_hash})
        await conn.commit()
        return True


async def revoke_user_refresh_tokens(user_id: str) -> int:
    """Revoke all refresh tokens for a user (security breach protocol)"""
    async_engine = get_async_engine()
    async with async_engine.begin() as conn:
        result = await conn.execute(text("""
            DELETE FROM refresh_tokens
            WHERE user_id = :user_id
        """), {"user_id": user_id})
        await conn.commit()
        return result.rowcount


async def create_refresh_token(user_id: str, ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> tuple[RefreshToken, str]:
    """Create and store a new refresh token"""
    refresh_token = secrets.token_urlsafe(256)
    token_hash = await hash_token(refresh_token)
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)

    token = RefreshToken(
        id=secrets.token_urlsafe(32),
        user_id=user_id,
        token_hash=token_hash,
        expires_at=expires_at,
        created_at=datetime.now(timezone.utc),
        ip_address=ip_address,
        user_agent=user_agent
    )

    async_engine = get_async_engine()
    async with async_engine.begin() as conn:
        await conn.execute(text("""
            INSERT INTO refresh_tokens (id, user_id, token_hash, expires_at, created_at, ip_address, user_agent)
            VALUES (:id, :user_id, :token_hash, :expires_at, :created_at, :ip_address, :user_agent)
        """), {
            "id": token.id,
            "user_id": token.user_id,
            "token_hash": token.token_hash,
            "expires_at": token.expires_at,
            "created_at": token.created_at,
            "ip_address": token.ip_address,
            "user_agent": token.user_agent
        })
        await conn.commit()
        return token, refresh_token


async def get_refresh_token_by_id(token_id: str) -> Optional[RefreshToken]:
    """Get a refresh token by its ID"""
    async_engine = get_async_engine()
    async with async_engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT * FROM refresh_tokens
            WHERE id = :token_id
        """), {"token_id": token_id})
        row = await conn.fetchfirst(result)

        if row:
            return RefreshToken(**row.dict())
        return None
