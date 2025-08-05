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
    UserResponseSchema, # UserResponseSchema 임포트 추가
    refresh_tokens_db
)
from core.config import settings, get_db
from core.models import User

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

class Token(BaseModel):
    access_token: str
    token_type: str
    refresh_token: Optional[str] = None

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class TokenData(BaseModel):
    id: Optional[int] = None
    username: Optional[str] = None
    role: Optional[str] = None

async def get_current_user(
    token: str = Depends(oauth2_scheme), 
    db: AsyncSession = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: int = payload.get("id")
        username: str = payload.get("username")
        role: str = payload.get("role")
        if user_id is None or username is None or role is None:
            raise credentials_exception
        token_data = TokenData(id=user_id, username=username, role=role)
    except JWTError:
        raise credentials_exception
    
    user = await get_user_by_username(db, token_data.username)
    
    if user is None:
        raise credentials_exception
    
    return user

@router.post("/register", summary="회원가입", response_model=UserResponseSchema) # response_model 변경
async def register(user_data: UserCreateSchema, db: AsyncSession = Depends(get_db)):
    new_user = await register_user(db, user_data)
    if new_user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Username or Email already registered"
        )
    
    # User 객체에서 UserResponseSchema에 필요한 필드를 추출하여 반환
    return UserResponseSchema(
        id=new_user.id,
        username=new_user.username,
        email=new_user.user_detail.email if new_user.user_detail else None,
        name=new_user.user_detail.name if new_user.user_detail else None,
        role=new_user.role,
        student_id=new_user.user_detail.student_id if new_user.user_detail else None,
        major=new_user.user_detail.major if new_user.user_detail else None
    )

@router.post("/login", summary="로그인", response_model=Token)
async def login(user_login: UserLoginSchema, db: AsyncSession = Depends(get_db)):
    authenticated_user = await authenticate_user(db, user_login.username, user_login.password)
    if not authenticated_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password"
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"id": authenticated_user.id, "username": authenticated_user.username, "role": authenticated_user.role}, expires_delta=access_token_expires
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
    if not delete_refresh_token(refresh_token_request.refresh_token):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired refresh token"
        )
    return {"message": "Logged out successfully"}

@router.post("/refresh_token", summary="액세스 토큰 갱신", response_model=Token)
async def refresh_access_token(refresh_token_request: RefreshTokenRequest, db: AsyncSession = Depends(get_db)):
    username = get_refresh_token_user(refresh_token_request.refresh_token)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token"
        )
    
    delete_refresh_token(refresh_token_request.refresh_token)

    user = await get_user_by_username(db, username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found for refresh token"
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    new_access_token = create_access_token(
        data={"id": user.id, "username": user.username, "role": user.role}, expires_delta=access_token_expires
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
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    if current_user.username != user_delete.username:
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

@router.get("/me", summary="현재 사용자 정보 가져오기 (보호된 라우트)", response_model=UserResponseSchema) # response_model 변경
async def read_users_me(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    user = await get_user_by_username(db, current_user.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    return UserResponseSchema(
        id=user.id,
        username=user.username,
        email=user.user_detail.email if user.user_detail else None,
        name=user.user_detail.name if user.user_detail else None,
        role=user.role,
        student_id=user.user_detail.student_id if user.user_detail else None,
        major=user.user_detail.major if user.user_detail else None
    )
