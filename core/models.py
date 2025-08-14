from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, BigInteger, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from core.config import Base
from sqlalchemy.dialects.postgresql import JSONB # JSONB 타입을 사용하려면 이 주석을 해제하세요.

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String, nullable=False, comment='교수, 학생')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user_detail = relationship("UserDetails", back_populates="user", uselist=False, cascade="all, delete-orphan")
    notices = relationship("Notices", back_populates="author")
    cpx_results = relationship("CpxResults", back_populates="student")
    cpx_evaluations_as_evaluator = relationship("CpxEvaluations", back_populates="evaluator")

class UserDetails(Base):
    __tablename__ = "user_details"

    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True, comment='사용자 고유 식별자')
    email = Column(String(255), unique=True, nullable=False, comment='이메일 (로그인, 인증 등에 활용)')
    name = Column(String(255), nullable=False, comment='사용자 이름')
    student_id = Column(String(50), comment='학번')
    major = Column(String(255), comment='전공')
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment='계정 생성 시간')
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment='정보 마지막 업데이트 시간')

    user = relationship("User", back_populates="user_detail")

class Notices(Base):
    __tablename__ = "notices"

    notice_id = Column(Integer, primary_key=True, index=True, comment='공지사항 고유 식별자')
    title = Column(String(255), nullable=False, comment='공지사항 제목')
    content = Column(Text, nullable=False, comment='공지사항 내용')
    important = Column(Boolean, default=False, comment='공지사항 중요 여부 (True면 중요 공지)')
    author_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment='작성자 ID (users 테이블의 user_id 참조)')
    view_count = Column(Integer, default=0, comment='조회수')
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment='공지사항 생성 시간')
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment='공지사항 마지막 업데이트 시간')

    author = relationship("User", back_populates="notices")
    attachments = relationship("Attachments", back_populates="notice", cascade="all, delete-orphan")

class Attachments(Base):
    __tablename__ = "attachments"

    attachment_id = Column(Integer, primary_key=True, index=True, comment='첨부파일 고유 식별자')
    notice_id = Column(Integer, ForeignKey("notices.notice_id", ondelete="CASCADE"), nullable=False, comment='어떤 공지사항에 속한 파일인지 나타내는 외래 키')
    original_filename = Column(String(255), nullable=False, comment='사용자가 업로드한 원본 파일명')
    stored_filename = Column(String(255), unique=True, nullable=False, comment='서버에 저장된 실제 파일명 (중복 방지를 위한 UUID 등 사용)')
    file_path = Column(String(500), nullable=False, comment='서버 내 파일의 저장 경로 (예: /uploads/notices/)')
    file_size = Column(BigInteger, nullable=False, comment='파일 크기 (바이트 단위)')
    file_type = Column(String(100), nullable=False, comment='파일 MIME 타입 (예: application/pdf, image/jpeg)')
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now(), comment='파일 업로드 시간')

    notice = relationship("Notices", back_populates="attachments")

class CpxResults(Base):
    __tablename__ = "cpx_results"

    result_id = Column(Integer, primary_key=True, index=True, comment='CPX 실습 결과 고유 식별자')
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment='실습을 실행한 학생 ID (users 테이블의 user_id 참조)')
    patient_name = Column(String(255), nullable=False, comment='환자 이름 또는 역할 (가상 환자 이름)')
    practice_date = Column(DateTime(timezone=True), server_default=func.now(), comment='실습 실행 날짜 및 시간')
    evaluation_status = Column(String(50), nullable=False, comment='평가 상태 (예: 평가 진행중, 평가 완료, 교수 피드백 완료, 오류)') 
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment='결과 기록 생성 시간')
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment='결과 기록 마지막 업데이트 시간')

    student = relationship("User", back_populates="cpx_results")
    cpx_detail = relationship("CpxDetails", back_populates="cpx_result", uselist=False)
    cpx_evaluation = relationship("CpxEvaluations", back_populates="cpx_result", uselist=False)

class CpxDetails(Base):
    __tablename__ = "cpx_details"

    detail_id = Column(Integer, primary_key=True, index=True, comment='CPX 상세 정보 고유 식별자')
    result_id = Column(Integer, ForeignKey("cpx_results.result_id"), unique=True, nullable=False, comment='cpx_results 테이블의 result_id를 참조하는 외래 키')
    memo = Column(Text, comment='실습 중 작성한 메모')
    system_evaluation_data = Column(JSONB, comment='CPX 실습에 대한 AI 평가 결과 및 상세 데이터 (JSON)')
    last_updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment='상세 정보 마지막 업데이트 시간')

    cpx_result = relationship("CpxResults", back_populates="cpx_detail")

class CpxEvaluations(Base):
    __tablename__ = "cpx_evaluations"

    evaluation_id = Column(Integer, primary_key=True, index=True, comment='평가 고유 식별자')
    result_id = Column(Integer, ForeignKey("cpx_results.result_id"), unique=True, nullable=False, comment='평가 대상 CPX 실습 결과 ID (cpx_results 테이블의 result_id 참조)')
    evaluator_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment='평가를 수행한 교수/관리자 ID (users 테이블의 user_id 참조)')
    overall_score = Column(Integer, comment='전체 점수 (예: 100점 만점)')
    detailed_feedback = Column(Text, comment='학생에게 전달될 종합 피드백 및 상세 평가 코멘트')
    evaluation_date = Column(DateTime(timezone=True), server_default=func.now(), comment='평가 수행 날짜 및 시간')
    evaluation_status = Column(String(50), nullable=False, comment='평가 처리 상태 (예: 피드백 대기중, 피드백 완료)')
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment='평가 기록 생성 시간')
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment='평가 기록 마지막 업데이트 시간')

    cpx_result = relationship("CpxResults", back_populates="cpx_evaluation")
    evaluator = relationship("User", back_populates="cpx_evaluations_as_evaluator")
