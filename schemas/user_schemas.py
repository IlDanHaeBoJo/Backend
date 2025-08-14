from pydantic import BaseModel
from typing import Optional

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

class PasswordChangeSchema(BaseModel):
    current_password: str
    new_password: str

class ForgotPasswordRequestSchema(BaseModel):
    username: str
    email: str

class VerifyCodeRequestSchema(BaseModel):
    email: str
    code: str

class UserResponseSchema(BaseModel):
    id: int
    username: str
    email: str
    name: str
    role: str
    student_id: Optional[str] = None
    major: Optional[str] = None

    class Config:
        from_attributes = True
