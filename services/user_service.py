import bcrypt
from typing import Optional, Dict
from datetime import datetime, timedelta
from jose import jwt, JWTError
import uuid # uuid 모듈 추가

from core.config import settings

# In-memory user storage for demonstration purposes
# In a real application, this would be a database
users_db: Dict[str, Dict[str, str]] = {}

# In-memory refresh token storage
# In a real application, this would be a persistent database (e.g., Redis)
refresh_tokens_db: Dict[str, Dict[str, str]] = {} # {refresh_token_id: {"username": "user", "expires": "datetime"}}

def hash_password(password: str) -> str:
    """Hashes a password using bcrypt."""
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    return hashed_password.decode('utf-8')

def verify_password(password: str, hashed_password: str) -> bool:
    """Verifies a password against a hashed password."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def register_user(username: str, password: str) -> Optional[Dict[str, str]]:
    """Registers a new user."""
    if username in users_db:
        return None  # User already exists
    
    hashed_pw = hash_password(password)
    users_db[username] = {"username": username, "hashed_password": hashed_pw}
    return {"username": username}

def authenticate_user(username: str, password: str) -> Optional[Dict[str, str]]:
    """Authenticates a user."""
    user = users_db.get(username)
    if user and verify_password(password, user["hashed_password"]):
        return {"username": username}
    return None

def delete_user(username: str) -> bool:
    """Deletes a user."""
    if username in users_db:
        del users_db[username]
        # 사용자가 삭제될 때 해당 사용자의 모든 리프레시 토큰도 삭제
        tokens_to_delete = [
            token_id for token_id, data in refresh_tokens_db.items() 
            if data["username"] == username
        ]
        for token_id in tokens_to_delete:
            del refresh_tokens_db[token_id]
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
    Creates a refresh token and stores it.
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
    Deletes a refresh token.
    """
    if refresh_token_id in refresh_tokens_db:
        del refresh_tokens_db[refresh_token_id]
        return True
    return False
