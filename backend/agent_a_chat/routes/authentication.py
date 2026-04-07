"""
Authentication module for Agent A Chat API.
Handles user registration, login, and JWT token management.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
from uuid import uuid4

from shared.settings import get_settings

router = APIRouter()

# === Configuration ===
SECRET_KEY = "your-super-secret-key"  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

logger = logging.getLogger(__name__)

# === Schemas ===


class User(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[str] = None

# === Helper Functions ===


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# === Auth Routes ===


@router.post("/register")
async def register(user: User, request: Request):
    """
    Register a new user.
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
                    VALUES ( % s, % s, % s)
                    """,
                    (new_user_id, user.email, hashed_password)
                )

            token = create_access_token(
                data={"sub": new_user_id}, expires_delta=timedelta(days=1))
            return {"access_token": token, "token_type": "bearer"}
    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/login")
async def login(user: User, request: Request):
    """
    Login user and return JWT token.
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

            token = create_access_token(
                data={"sub": str(row["id"])}, expires_delta=timedelta(days=1))
            return {"access_token": token, "token_type": "bearer"}
    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/me")
async def get_current_user(token: str = Depends(oauth2_scheme), request: Request = None):
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


def get_current_user_dict(token: str = Depends(oauth2_scheme)):
    """
    Get current authenticated user as dict for dependency injection.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"user_id": user_id}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
