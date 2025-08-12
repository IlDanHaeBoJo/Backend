from fastapi import APIRouter, HTTPException, Depends, status
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_db
from core.models import User
from schemas.user_schemas import UserResponseSchema
from services.user_service import get_user_by_username, get_user_by_id, get_all_users, delete_user
from routes.auth import get_current_user # get_current_user는 auth 라우터에서 가져옵니다.

router = APIRouter(prefix="/users", tags=["User Management"])

async def get_current_admin_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="관리자 권한이 필요합니다."
        )
    return current_user

@router.get("/me", summary="현재 사용자 정보 가져오기", response_model=UserResponseSchema)
async def read_users_me(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    user = await get_user_by_username(db, current_user.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="사용자를 찾을 수 없습니다."
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

@router.get("/students", summary="모든 학생 정보 가져오기 (관리자 전용)", response_model=List[UserResponseSchema])
async def get_all_students(
    current_admin_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    all_users = await get_all_users(db)
    students = [user for user in all_users if user.role == "student"]
    
    return [
        UserResponseSchema(
            id=student.id,
            username=student.username,
            email=student.user_detail.email if student.user_detail else None,
            name=student.user_detail.name if student.user_detail else None,
            role=student.role,
            student_id=student.user_detail.student_id if student.user_detail else None,
            major=student.user_detail.major if student.user_detail else None
        ) for student in students
    ]

@router.get("/students/{user_id}", summary="특정 학생 정보 가져오기 (관리자 전용)", response_model=UserResponseSchema)
async def get_student_by_id(
    user_id: int,
    current_admin_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    user = await get_user_by_id(db, user_id)
    if not user or user.role != "student":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="학생을 찾을 수 없습니다."
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

@router.delete("/students/{user_id}", summary="특정 학생 삭제 (관리자 전용)")
async def delete_student(
    user_id: int,
    current_admin_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    user_to_delete = await get_user_by_id(db, user_id)
    if not user_to_delete or user_to_delete.role != "student":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="삭제할 학생을 찾을 수 없습니다."
        )
    
    # delete_user 함수는 username을 인자로 받으므로, user_to_delete.username을 전달
    if not await delete_user(db, user_to_delete.username):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="학생 삭제에 실패했습니다."
        )
    
    return {"message": f"학생 (ID: {user_id})이 성공적으로 삭제되었습니다."}
