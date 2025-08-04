from typing import Optional, Dict
from datetime import datetime, timedelta
from jose import jwt, JWTError
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload # selectinload 임포트 추가
from passlib.context import CryptContext

from core.config import settings
from core.models import User, UserDetails # UserDetails 모델 임포트
from pydantic import BaseModel # Pydantic BaseModel 임포트

# In-memory refresh token storage (사용자 요청에 따라 유지)
refresh_tokens_db: Dict[str, Dict[str, str]] = {} # {refresh_token_id: {"username": "user", "expires": "datetime"}}

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserCreateSchema(BaseModel):
    username: str
    password: str
    email: str
    name: str
    role: str
    student_id: Optional[str] = None
    major: Optional[str] = None

class UserLoginSchema(BaseModel):
    username: str
    password: str

class UserDeleteSchema(BaseModel):
    username: str
    password: str

def hash_password(password: str) -> str:
    """Hashes a password using bcrypt."""
    return pwd_context.hash(password)

def verify_password(password: str, hashed_password: str) -> bool:
    """Verifies a password against a hashed password."""
    return pwd_context.verify(password, hashed_password)

async def register_user(db: AsyncSession, user_data: UserCreateSchema) -> Optional[User]:
    """Registers a new user in the database."""
    # Check if username or email already exists
    existing_user = await db.execute(select(User).where(User.username == user_data.username))
    if existing_user.scalar_one_or_none():
        return None # Username already exists

    existing_user_detail = await db.execute(select(UserDetails).where(UserDetails.email == user_data.email))
    if existing_user_detail.scalar_one_or_none():
        return None # Email already exists
    
    hashed_pw = hash_password(user_data.password)
    
    new_user = User(
        username=user_data.username,
        hashed_password=hashed_pw,
        role=user_data.role
    )
    db.add(new_user)
    await db.flush() # Flush to get the new_user.id

    new_user_detail = UserDetails(
        user_id=new_user.id,
        email=user_data.email,
        name=user_data.name,
        student_id=user_data.student_id,
        major=user_data.major
    )
    db.add(new_user_detail)
    await db.commit()
    await db.refresh(new_user)
    await db.refresh(new_user_detail)
    return new_user

async def authenticate_user(db: AsyncSession, username: str, password: str) -> Optional[User]:
    """Authenticates a user against the database."""
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    if user and verify_password(password, user.hashed_password):
        return user
    return None

async def delete_user(db: AsyncSession, username: str) -> bool:
    """Deletes a user and their associated details from the database."""
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    if user:
        # Delete associated user details
        await db.execute(UserDetails.__table__.delete().where(UserDetails.user_id == user.id))
        
        # Delete associated refresh tokens (if any, though managed in-memory)
        tokens_to_delete = [
            token_id for token_id, data in refresh_tokens_db.items() 
            if data["username"] == username
        ]
        for token_id in tokens_to_delete:
            del refresh_tokens_db[token_id]

        # Delete the user
        await db.delete(user)
        await db.commit()
        return True
    return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Creates a JWT access token.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def create_refresh_token(username: str, expires_delta: Optional[timedelta] = None) -> str:
    """
    Creates a refresh token and stores it in-memory.
    """
    refresh_token_id = str(uuid.uuid4())
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
    
    refresh_tokens_db[refresh_token_id] = {"username": username, "expires": expire}
    return refresh_token_id

def get_refresh_token_user(refresh_token_id: str) -> Optional[str]:
    """
    Retrieves the username from a refresh token ID, if valid and not expired.
    """
    token_data = refresh_tokens_db.get(refresh_token_id)
    if token_data and datetime.utcnow() < token_data["expires"]:
        return token_data["username"]
    return None

def delete_refresh_token(refresh_token_id: str) -> bool:
    """
    Deletes a refresh token from in-memory storage.
    """
    if refresh_token_id in refresh_tokens_db:
        del refresh_tokens_db[refresh_token_id]
        return True
    return False

async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    """Retrieves a user by username from the database, eagerly loading user_detail."""
    result = await db.execute(
        select(User)
        .options(selectinload(User.user_detail)) # user_detail 관계를 미리 로드
        .where(User.username == username)
    )
    print("result:",result)
    return result.scalar_one_or_none()
