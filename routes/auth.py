from fastapi import APIRouter, HTTPException, Depends, status, Response, Request
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional
from jose import jwt, JWTError
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from services.user_service import (
    register_user,
    authenticate_user,
    delete_user,
    create_access_token,
    create_refresh_token,
    get_refresh_token_user,
    delete_refresh_token,
    get_user_by_username,
    UserCreateSchema,
    UserLoginSchema,
    UserDeleteSchema,
    refresh_tokens_db
)
from core.config import settings, get_db
from core.models import User

router = APIRouter() # prefix와 tags 제거

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login") # tokenUrl 수정

class Token(BaseModel):
    access_token: str
    token_type: str
    refresh_token: Optional[str] = None

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class TokenData(BaseModel):
    id: Optional[int] = None # id 필드 추가
    username: Optional[str] = None # sub 대신 username으로 변경
    role: Optional[str] = None

async def get_current_user(
    token: str = Depends(oauth2_scheme), 
    db: AsyncSession = Depends(get_db)
) -> User:
    print("Current refresh_tokens_db (get_current_user):", refresh_tokens_db)
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        print("Decoded JWT payload:", payload)
        user_id: int = payload.get("id") # id로 가져옴
        username: str = payload.get("username") # username으로 가져옴
        role: str = payload.get("role")
        if user_id is None or username is None or role is None:
            raise credentials_exception
        token_data = TokenData(id=user_id, username=username, role=role) # id와 username 사용
        print("token data:",token_data)
    except JWTError:
        raise credentials_exception
    
    user = await get_user_by_username(db, token_data.username) # username으로 사용자 조회
    
    if user is None:
        raise credentials_exception
    
    print(f"User '{user.username}' loaded. UserDetail exists: {user.user_detail is not None}")
    if user.user_detail:
        print(f"UserDetail email: {user.user_detail.email}")
    
    return user

@router.post("/register", summary="회원가입", response_model=UserCreateSchema)
async def register(user_data: UserCreateSchema, db: AsyncSession = Depends(get_db)):
    print("Current refresh_tokens_db (register):", refresh_tokens_db)
    new_user = await register_user(db, user_data)
    if new_user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Username or Email already registered"
        )
    return user_data

@router.post("/login", summary="로그인", response_model=Token)
async def login(user_login: UserLoginSchema, db: AsyncSession = Depends(get_db)):
    print("Current refresh_tokens_db (login):", refresh_tokens_db)
    authenticated_user = await authenticate_user(db, user_login.username, user_login.password)
    if not authenticated_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password"
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"id": authenticated_user.id, "username": authenticated_user.username, "role": authenticated_user.role}, expires_delta=access_token_expires # id와 username 사용
    )
    
    refresh_token_expires = timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
    refresh_token = create_refresh_token(
        username=authenticated_user.username, expires_delta=refresh_token_expires
    )
    
    return {
        "access_token": access_token, 
        "token_type": "bearer", 
        "refresh_token": refresh_token
    }

@router.post("/logout", summary="로그아웃")
async def logout(refresh_token_request: RefreshTokenRequest):
    print("Current refresh_tokens_db (logout):", refresh_tokens_db)
    if not delete_refresh_token(refresh_token_request.refresh_token):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired refresh token"
        )
    return {"message": "Logged out successfully"}

@router.post("/refresh_token", summary="액세스 토큰 갱신", response_model=Token)
async def refresh_access_token(refresh_token_request: RefreshTokenRequest, db: AsyncSession = Depends(get_db)): # db 주입
    print("Current refresh_tokens_db (refresh_token):", refresh_tokens_db)
    username = get_refresh_token_user(refresh_token_request.refresh_token)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token"
        )
    
    delete_refresh_token(refresh_token_request.refresh_token)

    # username으로 User 객체 조회하여 id와 role 가져오기
    user = await get_user_by_username(db, username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found for refresh token"
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    new_access_token = create_access_token(
        data={"id": user.id, "username": user.username, "role": user.role}, expires_delta=access_token_expires # id와 username, role 사용
    )

    refresh_token_expires = timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
    new_refresh_token = create_refresh_token(
        username=username, expires_delta=refresh_token_expires
    )
    
    return {
        "access_token": new_access_token, 
        "token_type": "bearer", 
        "refresh_token": new_refresh_token
    }

@router.delete("/delete_account", summary="회원 탈퇴")
async def delete_account(
    user_delete: UserDeleteSchema,
    current_user: User = Depends(get_current_user), # User 객체로 유지
    db: AsyncSession = Depends(get_db)
):
    print("Current refresh_tokens_db (delete_account):", refresh_tokens_db)
    if current_user.username != user_delete.username: # username 비교
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Cannot delete another user's account"
        )
    
    authenticated = await authenticate_user(db, user_delete.username, user_delete.password)
    if not authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect password"
        )

    if not await delete_user(db, user_delete.username):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    
    return {"message": "Account deleted successfully"}

@router.get("/me", summary="현재 사용자 정보 가져오기 (보호된 라우트)", response_model=UserCreateSchema)
async def read_users_me(
    current_user: User = Depends(get_current_user), # User 객체로 유지
    db: AsyncSession = Depends(get_db) # db 주입
):
    # username으로 User 객체 조회
    user = await get_user_by_username(db, current_user.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    print("Current refresh_tokens_db (read_users_me):", refresh_tokens_db)
    if not user.user_detail:
        print(f"Warning: User '{user.username}' has no associated UserDetails.")
    
    return UserCreateSchema(
        id=user.id,
        username=user.username,
        password="[PROTECTED]",
        email=user.user_detail.email if user.user_detail else None,
        name=user.user_detail.name if user.user_detail else None,
        role=user.role,
        student_id=user.user_detail.student_id if user.user_detail else None,
        major=user.user_detail.major if user.user_detail else None
    )
