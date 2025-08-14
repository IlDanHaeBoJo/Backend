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
    get_user_by_email, # get_user_by_email 임포트 추가
    update_password, # update_password 임포트 추가
    generate_temporary_password, # generate_temporary_password 임포트 추가
    send_temporary_password_email, # send_temporary_password_email 임포트 추가
    generate_verification_code, # generate_verification_code 임포트 추가
    send_verification_email, # send_verification_email 임포트 추가
    verify_code, # verify_code 임포트 추가
    verify_password,
    refresh_tokens_db,
    verification_codes_db # verification_codes_db 임포트 추가
)
from schemas.user_schemas import ( # schemas/user_schemas.py에서 스키마 임포트
    UserCreateSchema,
    UserLoginSchema,
    UserDeleteSchema,
    PasswordChangeSchema,
    ForgotPasswordRequestSchema,
    VerifyCodeRequestSchema,
    UserResponseSchema
)
from core.config import settings, get_db
from core.models import User
from datetime import datetime, timedelta # datetime, timedelta 임포트 추가

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
        email=user_data.email, # user_data에서 직접 이메일 사용
        name=user_data.name, # user_data에서 직접 이름 사용
        role=new_user.role,
        student_id=user_data.student_id, # user_data에서 직접 학번 사용
        major=user_data.major # user_data에서 직접 전공 사용
    )

@router.patch("/change-password", summary="비밀번호 변경")
async def change_password(
    password_change: PasswordChangeSchema,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    if not verify_password(password_change.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect current password"
        )
    
    await update_password(db, current_user, password_change.new_password)
    return {"message": "Password updated successfully"}

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

@router.post("/request-password-reset", summary="비밀번호 재설정 요청 (본인 확인 코드 전송)")
async def request_password_reset(
    request_data: ForgotPasswordRequestSchema,
    db: AsyncSession = Depends(get_db)
):
    user = await get_user_by_username(db, request_data.username)
    if not user or (user.user_detail and user.user_detail.email != request_data.email):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="User with provided username and email not found"
        )
    
    # 본인 확인 코드 생성 및 저장
    code = generate_verification_code()
    expires_at = datetime.utcnow() + timedelta(minutes=settings.VERIFICATION_CODE_EXPIRE_MINUTES) # 설정 파일에 추가 필요
    verification_codes_db[request_data.email] = {"code": code, "expires": expires_at}

    # 이메일 전송 (실제 구현 필요)
    await send_verification_email(request_data.email, code)

    return {"message": "Verification code sent to your email"}

@router.post("/verify-password-reset-code", summary="비밀번호 재설정 본인 확인 코드 검증 및 임시 비밀번호 발급")
async def verify_password_reset_code(
    verify_data: VerifyCodeRequestSchema,
    db: AsyncSession = Depends(get_db) # DB 세션 추가
):
    if not verify_code(verify_data.email, verify_data.code):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Invalid or expired verification code"
        )
    
    # 본인 확인이 성공하면 해당 이메일로 사용자 조회
    user = await get_user_by_email(db, verify_data.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="User not found for the provided email"
        )
    
    # 임시 비밀번호 생성 (아직 DB에 업데이트하지 않음)
    temp_password = await generate_temporary_password()

    # 임시 비밀번호 이메일 전송
    try:
        await send_temporary_password_email(verify_data.email, temp_password)
        # 이메일 전송 성공 시에만 비밀번호 업데이트
        await update_password(db, user, temp_password)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send temporary password email: {e}"
        )

    return {"message": "Verification successful. Temporary password sent to your email."}
