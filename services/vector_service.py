import os
import logging
from typing import List, Optional
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)

# SQLite 버전 업그레이드를 위해 pysqlite3을 sqlite3로 대체
try:
    import pysqlite3 as sqlite3
    import sys
    sys.modules['sqlite3'] = sqlite3
    logger.info("✅ pysqlite3를 사용하여 SQLite 버전 업그레이드")
except ImportError:
    import sqlite3
    logger.warning("⚠️  pysqlite3 없음, 기본 sqlite3 사용")

try:
    import chromadb
    from chromadb.config import Settings
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import Chroma
    from langchain.document_loaders import TextLoader, DirectoryLoader
    CHROMADB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ChromaDB import 실패: {e}")
    CHROMADB_AVAILABLE = False
except RuntimeError as e:
    if "sqlite3" in str(e).lower():
        logger.warning(f"SQLite 버전 이슈로 ChromaDB 비활성화: {e}")
        CHROMADB_AVAILABLE = False
    else:
        logger.error(f"ChromaDB 런타임 오류: {e}")
        CHROMADB_AVAILABLE = False
except Exception as e:
    logger.error(f"ChromaDB 초기화 오류: {e}")
    CHROMADB_AVAILABLE = False

class VectorService:
    def __init__(self):
        """벡터 서비스 초기화"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        self.persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        self._use_fallback_knowledge = False
        
        if CHROMADB_AVAILABLE:
            try:
                # OpenAI 임베딩 초기화
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=self.api_key,
                    model="text-embedding-ada-002"
                )
                
                # ChromaDB 클라이언트 초기화 (SQLite 오류 처리)
                self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
                
                # Langchain Chroma 벡터스토어 초기화
                self.vectorstore = Chroma(
                    client=self.chroma_client,
                    collection_name="cpx_cases",
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )
                
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                
                logger.info("✅ Vector 서비스 초기화 완료")
                
                # 초기 데이터 로드 (필요한 경우)
                asyncio.create_task(self._initialize_knowledge_base())
                
            except RuntimeError as e:
                if "sqlite3" in str(e).lower():
                    logger.warning("⚠️  SQLite 버전 이슈 감지 - pysqlite3 설치 후 재시도 권장")
                    logger.warning(f"   오류 상세: {e}")
                    logger.info("   해결 방법: pip install pysqlite3-binary")
                    self.vectorstore = None
                    self._use_fallback_knowledge = True
                else:
                    logger.error(f"❌ Vector 서비스 초기화 실패: {e}")
                    self.vectorstore = None
                    self._use_fallback_knowledge = True
            except Exception as e:
                logger.error(f"❌ Vector 서비스 초기화 실패: {e}")
                self.vectorstore = None
                self._use_fallback_knowledge = True
        else:
            logger.warning("⚠️  ChromaDB가 설치되지 않음, 기본 응답 모드로 동작")
            self.vectorstore = None
            self._use_fallback_knowledge = True

    async def _initialize_knowledge_base(self):
        """지식 베이스 초기 데이터 로드"""
        try:
            # 기존 데이터가 있는지 확인
            collection_count = self.vectorstore._collection.count()
            if collection_count > 0:
                logger.info(f"기존 벡터 DB에 {collection_count}개 문서가 있습니다.")
                return
            
                            # 초기 CPX 케이스 데이터 추가
            initial_docs = [
                {
                    "content": """
                    CPX 케이스 1 - 내과: 급성 위염
                    환자 정보: 김민수, 35세 남성, 사무직
                    주증상: 속쓰림, 구토, 복통
                    병력: 어제 저녁 회식 후 급성 증상 발생, 음주력 있음
                    현재 상태: 명치 부위 통증 호소, 구토 2회, 식은땀
                    환자 반응: "어제부터 속이 너무 아파요. 토하고 나서 조금 나아졌는데 계속 쓰려요."
                    """,
                    "metadata": {"category": "internal_medicine", "case_id": "IM_001", "department": "내과"}
                },
                {
                    "content": """
                    CPX 케이스 2 - 정신과: 우울증
                    환자 정보: 이지은, 28세 여성, 대학원생
                    주증상: 우울감, 불면, 식욕저하, 집중력 감소
                    병력: 2개월 전부터 증상 시작, 최근 학업 스트레스 증가
                    현재 상태: 무기력, 흥미 상실, 체중 감소 5kg
                    환자 반응: "요즘 아무것도 하기 싫고 잠도 안 와요. 매일 우울해요."
                    """,
                    "metadata": {"category": "psychiatry", "case_id": "PSY_001", "department": "정신과"}
                },
                {
                    "content": """
                    CPX 케이스 3 - 외과: 충수염 의심
                    환자 정보: 박준호, 22세 남성, 대학생
                    주증상: 우하복부 통증, 발열, 구토
                    병력: 오늘 새벽부터 급성 복통 시작, 통증이 점차 우하복부로 이동
                    현재 상태: 맥브니 압점 압통, 38.5도 발열, 보행 시 통증 증가
                    환자 반응: "새벽에 갑자기 배가 아프기 시작했어요. 걸을 때 더 아파요."
                    """,
                    "metadata": {"category": "surgery", "case_id": "SUR_001", "department": "외과"}
                },
                # 한국어 의료 용어 매핑 데이터베이스 추가
                {
                    "content": """
                    한국어 의료 용어 매핑 - 소화기계
                    일반 표현 → 의학 용어:
                    "속쓰림, 속이 아파요" → "위산 과다, 위염"
                    "토할 것 같아요, 구역질" → "오심, 구토"
                    "배가 아파요, 복통" → "복부 통증, 복부 불편감"
                    "소화가 안돼요" → "소화불량, 위장장애"
                    "속이 더부룩해요" → "복부 팽만감"
                    "가슴이 아파요" → "흉통, 심장 관련 통증"
                    "숨이 차요" → "호흡곤란, 숨참"
                    """,
                    "metadata": {"category": "terminology", "case_id": "TERM_001", "department": "소화기내과"}
                },
                {
                    "content": """
                    한국어 의료 용어 매핑 - 신경정신과
                    일반 표현 → 의학 용어:
                    "머리가 아파요" → "두통, 편두통"
                    "어지러워요" → "현기증, 어지럼증"
                    "잠이 안와요" → "불면증, 수면장애"
                    "우울해요, 기분이 안좋아요" → "우울감, 우울증상"
                    "불안해요, 걱정돼요" → "불안감, 불안장애"
                    "기억이 안나요" → "기억상실, 인지장애"
                    "집중이 안돼요" → "집중력 저하, 주의력 결핍"
                    """,
                    "metadata": {"category": "terminology", "case_id": "TERM_002", "department": "신경정신과"}
                },
                {
                    "content": """
                    한국어 CPX 질문 패턴 - 병력 청취
                    학생 질문 예시:
                    "언제부터 아프셨나요?" → 발병 시기 문진
                    "어떻게 아프신가요?" → 증상 양상 파악
                    "어디가 아프신가요?" → 부위별 통증 확인
                    "얼마나 아파요?" → 통증 정도 평가
                    "다른 증상은 없으신가요?" → 동반 증상 확인
                    "가족력이 있으신가요?" → 가족병력 문진
                    "복용하시는 약이 있나요?" → 약물력 확인
                    "알레르기가 있으신가요?" → 알레르기 병력
                    """,
                    "metadata": {"category": "questioning", "case_id": "Q_001", "department": "일반"}
                },
                {
                    "content": """
                    한국어 환자 반응 패턴 - 자연스러운 응답
                    통증 표현:
                    "아야, 아프다", "시큰시큰해요", "찌릿찌릿해요"
                    "쿡쿡 쑤셔요", "욱신욱신해요", "뻐근해요"
                    
                    시간 표현:
                    "어제부터", "3일 전부터", "한 일주일 정도"
                    "새벽에", "점심 먹고 나서", "밤에 자려고 누우면"
                    
                    정도 표현:
                    "조금", "좀", "많이", "너무", "참을 수 없을 정도로"
                    "견딜 만해요", "심하지는 않아요", "갈수록 심해져요"
                    """,
                    "metadata": {"category": "patient_response", "case_id": "PR_001", "department": "일반"}
                }
            ]
            
            # 문서 분할 및 벡터 DB에 추가
            for doc_data in initial_docs:
                texts = self.text_splitter.split_text(doc_data["content"])
                metadatas = [doc_data["metadata"] for _ in texts]
                
                self.vectorstore.add_texts(
                    texts=texts,
                    metadatas=metadatas
                )
            
            logger.info("초기 지식 베이스 데이터 로드 완료")
            
        except Exception as e:
            logger.error(f"지식 베이스 초기화 실패: {e}")

    async def search(self, query: str, k: int = 3) -> List[str]:
        """쿼리와 관련된 문서 검색"""
        if not self.vectorstore:
            logger.info("벡터 검색 비활성화됨, 기본 의료 지식 사용")
            return await self._get_fallback_knowledge(query)
        
        try:
            # 유사도 검색
            docs = self.vectorstore.similarity_search(query, k=k)
            
            # 문서 내용 추출
            relevant_texts = [doc.page_content for doc in docs]
            
            logger.info(f"벡터 검색 완료: {len(relevant_texts)}개 문서 반환")
            return relevant_texts
            
        except Exception as e:
            logger.error(f"벡터 검색 실패, 기본 지식 사용: {e}")
            return await self._get_fallback_knowledge(query)
    
    async def _get_fallback_knowledge(self, query: str) -> List[str]:
        """벡터 검색 실패 시 기본 의료 지식 반환"""
        query_lower = query.lower()
        
        # 증상별 기본 CPX 케이스 매핑
        fallback_knowledge = []
        
        if any(symptom in query_lower for symptom in ["속", "위", "배", "복통", "구토", "토"]):
            fallback_knowledge.append("""
            CPX 케이스 - 급성 위염
            환자: 35세 남성, 회식 후 급성 증상
            주증상: 속쓰림, 구토, 명치 부위 통증
            환자 표현: "어제부터 속이 너무 아파요. 토하고 나서 조금 나아졌는데 계속 쓰려요."
            """)
        
        if any(symptom in query_lower for symptom in ["우울", "기분", "잠", "불면", "무기력"]):
            fallback_knowledge.append("""
            CPX 케이스 - 우울증  
            환자: 28세 여성, 대학원생
            주증상: 우울감, 불면, 식욕저하, 집중력 감소
            환자 표현: "요즘 아무것도 하기 싫고 잠도 안 와요. 매일 우울해요."
            """)
            
        if any(symptom in query_lower for symptom in ["열", "감기", "기침", "목"]):
            fallback_knowledge.append("""
            CPX 케이스 - 상기도 감염
            환자: 25세 여성
            주증상: 발열, 기침, 인후통
            환자 표현: "3일 전부터 열이 나고 목이 아파요. 기침도 계속 나요."
            """)
        
        if not fallback_knowledge:
            # 기본 환자 정보
            fallback_knowledge.append("""
            CPX 일반 환자 정보
            환자는 증상에 대해 솔직하고 자세히 답변합니다.
            의학 지식이 없는 일반인 관점에서 증상을 일상 언어로 표현합니다.
            """)
        
        logger.info(f"기본 지식 반환: {len(fallback_knowledge)}개 항목")
        return fallback_knowledge

    async def add_document(self, content: str, metadata: dict = None) -> bool:
        """새 문서를 벡터 DB에 추가"""
        if not self.vectorstore:
            return False
        
        try:
            # 텍스트 분할
            texts = self.text_splitter.split_text(content)
            metadatas = [metadata or {} for _ in texts]
            
            # 벡터 DB에 추가
            self.vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas
            )
            
            logger.info(f"문서 추가 완료: {len(texts)}개 청크")
            return True
            
        except Exception as e:
            logger.error(f"문서 추가 실패: {e}")
            return False

    async def add_documents_from_directory(self, directory_path: str) -> bool:
        """디렉토리의 모든 텍스트 파일을 벡터 DB에 추가"""
        if not self.vectorstore:
            return False
        
        try:
            # 디렉토리 로더 사용
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            
            documents = loader.load()
            
            if not documents:
                logger.warning(f"디렉토리에서 문서를 찾을 수 없음: {directory_path}")
                return False
            
            # 텍스트 분할
            texts = self.text_splitter.split_documents(documents)
            
            # 벡터 DB에 추가
            self.vectorstore.add_documents(texts)
            
            logger.info(f"디렉토리 문서 추가 완료: {len(texts)}개 청크")
            return True
            
        except Exception as e:
            logger.error(f"디렉토리 문서 추가 실패: {e}")
            return False

    def get_collection_stats(self) -> dict:
        """벡터 DB 통계 정보 반환"""
        if not self.vectorstore:
            return {"error": "벡터스토어가 초기화되지 않음"}
        
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "document_count": count,
                "collection_name": collection.name,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"통계 정보 조회 실패: {e}")
            return {"error": str(e)}

    async def clear_collection(self) -> bool:
        """벡터 DB 컬렉션 초기화"""
        if not self.vectorstore:
            return False
        
        try:
            # 컬렉션 삭제
            self.chroma_client.delete_collection("cpx_cases")
            
            # 새 컬렉션 생성
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name="cpx_cases",
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            logger.info("벡터 DB 컬렉션 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"컬렉션 초기화 실패: {e}")
            return False 