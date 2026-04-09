"""
Authentication module for Agent A Chat API.
Handles user registration, login, and JWT token management.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
from uuid import uuid4

from shared.settings import get_settings
from entities.token import (
    create_token_pair,
    validate_refresh_token,
    revoke_refresh_token,
    revoke_user_refresh_tokens,
)
from entities.database import User

router = APIRouter()

# === Configuration ===
SECRET_KEY = get_settings().secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15  # Updated to 15 minutes

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

logger = logging.getLogger(__name__)

# === Schemas ===


class User(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[str] = None


class RefreshTokenRequest(BaseModel):
    refresh_token: str

# === Helper Functions ===


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(
        timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# === Auth Routes ===


@router.post("/register")
async def register(user: User, request: Request):
    """
    Register a new user and issue tokens.
    """
    try:
        db_pool = request.app.state.db_pool

        async with db_pool.connection() as conn:
            result = await conn.execute(
                "SELECT id FROM users WHERE email = %s",
                (user.email,)
            )
            if await result.fetchone():
                raise HTTPException(
                    status_code=400, detail="Email already registered")

            hashed_password = get_password_hash(user.password)
            new_user_id = str(uuid4())

            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO users(id, email, password_hash)
                    VALUES (%s, %s, %s)
                    """,
                    (new_user_id, user.email, hashed_password)
                )

            # Create token pair (access + refresh)
            access_token, refresh_token = await create_token_pair(new_user_id)

            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer"
            }
    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/login")
async def login(user: User, request: Request):
    """
    Login user and issue tokens.
    """
    try:
        db_pool = request.app.state.db_pool

        async with db_pool.connection() as conn:
            result = await conn.execute(
                "SELECT id, email, password_hash FROM users WHERE email = %s",
                (user.email,)
            )
            row = await result.fetchone()
            if not row:
                raise HTTPException(
                    status_code=400, detail="Invalid credentials")

            if not verify_password(user.password, row["password_hash"]):
                raise HTTPException(
                    status_code=400, detail="Invalid credentials")

            # Create token pair (access + refresh)
            access_token, refresh_token = await create_token_pair(str(row["id"]))

            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer"
            }
    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refresh")
async def refresh_tokens(refresh_token_request: RefreshTokenRequest, request: Request):
    """
    Refresh access token using refresh token.
    Implements token rotation and security breach detection.
    """
    try:
        # Validate refresh token against DB
        rt = await validate_refresh_token(refresh_token_request.refresh_token, request)
        if not rt:
            raise HTTPException(
                status_code=401, detail="Invalid or expired refresh token"
            )

        # Check for token reuse (security breach)
        if rt.used_at is not None and rt.used_at != datetime.now(timezone.utc):
            logger.warning(f"Token reuse detected for user {rt.user_id}")
            # Revoke all tokens for this user (security breach protocol)
            revoked_count = await revoke_user_refresh_tokens(rt.user_id)
            logger.warning(
                f"Revoked {revoked_count} tokens for user {rt.user_id}")
            raise HTTPException(
                status_code=401, detail="Security breach: token reuse detected"
            )

        # Create new token pair (rotation) with updated IP/user_agent
        new_access_token, new_refresh_token = await create_token_pair(
            str(rt.user_id),
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )

        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Refresh error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/logout")
async def logout(refresh_token: str = None, request: Request = None):
    """
    Logout user and revoke refresh token.
    """
    try:
        if refresh_token:
            await revoke_refresh_token(refresh_token)
        # If called without refresh token (e.g., from client), just log out
        logger.info("User logged out")
        return {"message": "Logged out successfully"}
    except Exception as e:
        logger.error(f"Logout error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/me")
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Get current authenticated user.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"user_id": user_id}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
