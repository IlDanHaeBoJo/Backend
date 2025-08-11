"""
공통 상수 정의
"""

# 파일 관련 상수
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_FILE_SIZE_MB = 10

# 허용된 파일 타입
ALLOWED_FILE_TYPES = [
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-powerpoint',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    'image/jpeg',
    'image/png',
    'image/gif',
    'text/plain'
]

# S3 관련 상수
S3_PRESIGNED_URL_EXPIRES_IN = 3600  # 1시간

# 권한 관련 상수
ADMIN_ROLES = ["admin", "교수"]

# 에러 메시지
ERROR_MESSAGES = {
    "NOTICE_NOT_FOUND": "공지사항을 찾을 수 없습니다.",
    "ATTACHMENT_NOT_FOUND": "첨부파일을 찾을 수 없습니다.",
    "FILE_SIZE_EXCEEDED": f"파일 크기는 {MAX_FILE_SIZE_MB}MB를 초과할 수 없습니다.",
    "UNSUPPORTED_FILE_TYPE": "지원하지 않는 파일 형식입니다.",
    "S3_UPLOAD_FAILED": "S3 업로드 실패: {}",
    "DOWNLOAD_URL_GENERATION_FAILED": "다운로드 URL 생성에 실패했습니다.",
    "FILE_NOT_EXISTS": "파일이 존재하지 않습니다.",
    "PERMISSION_DENIED": "{} 권한이 없습니다.",
    "GENERAL_ERROR": "{} 중 오류가 발생했습니다."
}

