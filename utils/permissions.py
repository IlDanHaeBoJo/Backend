from functools import wraps
from fastapi import HTTPException, status, Depends
from typing import Callable
import logging

logger = logging.getLogger(__name__)

def require_role(required_role: str):
    """특정 역할이 필요한 엔드포인트를 위한 데코레이터"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 여기서는 간단한 구현을 위해 current_user를 확인
            # 실제로는 JWT 토큰에서 사용자 역할을 추출해야 함
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="인증이 필요합니다."
                )
            
            # 실제 구현에서는 사용자 역할을 확인해야 함
            # JWT 토큰에서 추출된 사용자 객체의 role 속성 확인
            if not hasattr(current_user, 'role'):
                logger.warning(f"User object {current_user} has no 'role' attribute.")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="사용자 역할 정보를 찾을 수 없습니다."
                )

            if current_user.role != required_role:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"{required_role} 권한이 필요합니다."
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
