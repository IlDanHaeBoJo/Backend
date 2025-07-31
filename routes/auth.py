from fastapi import APIRouter, HTTPException, Depends, status, Response, Request
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional
from jose import jwt, JWTError
from datetime import datetime, timedelta

from services.user_service import (
    register_user,
    authenticate_user,
    delete_user,
    create_access_token,
    create_refresh_token,
    get_refresh_token_user,
    delete_refresh_token,
    users_db, # users_db를 임포트하여 직접 접근 디버깅용
    refresh_tokens_db # refresh_tokens_db도 임포트하여 직접 접근 디버깅용
)
from core.config import settings

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserDelete(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    refresh_token: Optional[str] = None

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Dependency to get current user (for protected routes)
async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    print("Current users_db (get_current_user):", users_db) # 디버깅용
    print("Current refresh_tokens_db (get_current_user):", refresh_tokens_db) # 디버깅용
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    if token_data.username not in users_db:
        raise credentials_exception
        
    return token_data.username

@router.post("/register", summary="회원가입")
async def register(user: UserCreate):
    print("Current users_db (register):", users_db) # 디버깅용
    print("Current refresh_tokens_db (register):", refresh_tokens_db) # 디버깅용
    new_user = register_user(user.username, user.password)
    if new_user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered"
        )
    return {"message": "User registered successfully", "username": new_user["username"]}

@router.post("/login", summary="로그인", response_model=Token)
async def login(user: UserLogin):
    print("Current users_db (login):", users_db) # 디버깅용
    print("Current refresh_tokens_db (login):", refresh_tokens_db) # 디버깅용
    authenticated_user = authenticate_user(user.username, user.password)
    if not authenticated_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password"
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": authenticated_user["username"]}, expires_delta=access_token_expires
    )
    
    refresh_token_expires = timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
    refresh_token = create_refresh_token(
        username=authenticated_user["username"], expires_delta=refresh_token_expires
    )
    
    return {
        "access_token": access_token, 
        "token_type": "bearer", 
        "refresh_token": refresh_token
    }

@router.post("/logout", summary="로그아웃")
async def logout(refresh_token_request: RefreshTokenRequest):
    print("Current users_db (logout):", users_db) # 디버깅용
    print("Current refresh_tokens_db (logout):", refresh_tokens_db) # 디버깅용
    if not delete_refresh_token(refresh_token_request.refresh_token):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired refresh token"
        )
    return {"message": "Logged out successfully"}

@router.post("/refresh_token", summary="액세스 토큰 갱신", response_model=Token)
async def refresh_access_token(refresh_token_request: RefreshTokenRequest):
    print("Current users_db (refresh_token):", users_db) # 디버깅용
    print("Current refresh_tokens_db (refresh_token):", refresh_tokens_db) # 디버깅용
    username = get_refresh_token_user(refresh_token_request.refresh_token)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token"
        )
    
    # 기존 리프레시 토큰 삭제 (일회용 리프레시 토큰 구현)
    delete_refresh_token(refresh_token_request.refresh_token)

    # 새 액세스 토큰 생성
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    new_access_token = create_access_token(
        data={"sub": username}, expires_delta=access_token_expires
    )

    # 새 리프레시 토큰 생성 (선택 사항: 리프레시 토큰도 갱신)
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
async def delete_account(user: UserDelete, current_user: str = Depends(get_current_user)):
    print("Current users_db (delete_account):", users_db) # 디버깅용
    print("Current refresh_tokens_db (delete_account):", refresh_tokens_db) # 디버깅용
    if current_user != user.username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Cannot delete another user's account"
        )
    
    # Re-authenticate before deleting
    authenticated = authenticate_user(user.username, user.password)
    if not authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect password"
        )

    if not delete_user(user.username):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    
    return {"message": "Account deleted successfully"}

@router.get("/me", summary="현재 사용자 정보 가져오기 (보호된 라우트)")
async def read_users_me(current_user: str = Depends(get_current_user)):
    print("Current users_db (read_users_me):", users_db) # 디버깅용
    print("Current refresh_tokens_db (read_users_me):", refresh_tokens_db) # 디버깅용
    return {"username": current_user}
