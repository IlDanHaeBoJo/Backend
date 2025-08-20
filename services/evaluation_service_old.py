from typing import Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
from pathlib import Path
import json
import asyncio
import aiofiles
import logging
import os
import sys
import re
from collections import Counter

# LangGraph 관련 import
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage as AnyMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# CPX 관련 import
from services.cpx_service import CpxService
from core.config import get_db

# RAG 가이드라인 import
from RAG.guideline_retriever import GuidelineRetriever

# CPX 평가 상태 정의 (Multi-Step Reasoning 전용)
class CPXEvaluationState(TypedDict):
    """CPX 평가 상태 정의 - Multi-Step Reasoning 전용"""
    # 입력 데이터
    user_id: str
    scenario_id: str
    conversation_log: List[Dict]
    
    # Multi-Step Reasoning 결과들 (핵심)
    medical_context_analysis: Optional[Dict]
    question_intent_analysis: Optional[Dict]
    completeness_assessment: Optional[Dict]
    quality_evaluation: Optional[Dict]
    appropriateness_validation: Optional[Dict]
    
    # 종합 평가 결과
    comprehensive_evaluation: Optional[Dict]
    
    # 최종 결과
    final_scores: Optional[Dict]
    feedback: Optional[Dict]
    
    # 메타데이터
    evaluation_metadata: Optional[Dict]
    
    # 메시지 추적
    messages: Annotated[List[AnyMessage], add_messages]

class EvaluationService:
    def __init__(self):
        """CPX 평가 서비스 초기화"""
        # 기존 하드코딩된 시나리오 정보는 제거하고 JSON 기반으로 통합
        
        self.session_data = {}  # 세션별 평가 데이터
        
        # 평가 결과 저장 디렉터리
        self.evaluation_dir = Path("evaluation_results")
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        
        # LangGraph 기반 텍스트 평가 관련
        self.llm = None
        self.workflow = None
        self._initialize_langgraph_components()
        
        # RAG 기반 가이드라인 검색기 초기화
        self.guideline_retriever = None
        self._initialize_guideline_retriever()
        
        # 시나리오별 적용 요소들 정의
        self.scenario_applicable_elements = self._initialize_scenario_elements()

    def _initialize_guideline_retriever(self):
        """RAG 기반 가이드라인 검색기 초기화"""
        try:
            # 가이드라인 검색기 초기화
            rag_path = Path(__file__).parent.parent / "RAG"
            index_path = rag_path / "faiss_guideline_index"
            self.guideline_retriever = GuidelineRetriever(index_path=str(index_path))
            
            if self.guideline_retriever.vectorstore:
                print("✅ RAG 가이드라인 검색기 초기화 완료")
            else:
                print("⚠️ RAG 가이드라인 검색기 초기화 실패")
                self.guideline_retriever = None
                
        except Exception as e:
            print(f"❌ 가이드라인 검색기 초기화 오류: {e}")
            self.guideline_retriever = None

    def _initialize_scenario_elements(self) -> Dict:
        """시나리오별 적용 요소들 초기화 - scenarios/ 디렉토리에서 로드"""
        scenario_elements = {}
        scenario_dir = Path("scenarios")
        
        if not scenario_dir.exists():
            print("⚠️ scenarios 디렉토리가 없습니다.")
            return {}
        
        for json_file in scenario_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                scenario_info = data.get("scenario_info", {})
                category = scenario_info.get("category", "unknown")
                
                # 시나리오 ID를 키로 사용 (예: memory_impairment)
                scenario_id = category.replace(" ", "_").lower()
                
                scenario_elements[scenario_id] = {
                    "name": category,
                    "description": scenario_info.get("case_presentation", ""),
                    "patient_name": scenario_info.get("patient_name", ""),
                    "primary_diagnosis": scenario_info.get("primary_diagnosis", ""),
                    "differential_diagnoses": scenario_info.get("differential_diagnoses", []),
                    "applicable_areas": [
                        "history_taking",
                        "physical_examination", 
                        "patient_education"
                    ]
                }
                
                print(f"✅ 시나리오 로드: {category} ({json_file.name})")
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON 파싱 오류 ({json_file.name}): {e}")
            except Exception as e:
                print(f"❌ 시나리오 로드 오류 ({json_file.name}): {e}")
        
        return scenario_elements

    def _get_scenario_category(self, scenario_id: str) -> Optional[str]:
        """시나리오 파일에서 카테고리 정보 로드"""
        try:
            scenario_path = Path(f"scenarios/neurology_dementia_case.json")  # 현재는 하나의 시나리오만
            if not scenario_path.exists():
                return None
            
            with open(scenario_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("scenario_info", {}).get("category")
        except Exception as e:
            print(f"❌ 시나리오 카테고리 로드 실패: {e}")
            return None

    def _parse_structured_sections(self, document) -> dict:
        """문서에서 구조화된 섹션 파싱"""
        import re
        
        structured_sections = {}
        
        # 문서를 문자열로 변환
        if hasattr(document, 'page_content'):
            document_text = document.page_content
        elif isinstance(document, dict):
            document_text = document.get('content', '') or document.get('page_content', '') or str(document)
        else:
            document_text = str(document)
        
        # 섹션 패턴: 【섹션명】
        section_pattern = re.compile(r'【([^】]+)】')
        # 항목 패턴: • 또는 - 로 시작하는 줄
        bullet_pattern = re.compile(r'^\s*[•\-\*]\s+(.+)$', re.MULTILINE)
        
        sections = section_pattern.split(document_text)
        
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                section_name = sections[i].strip()
                section_content = sections[i + 1]
                
                # 섹션 내용에서 필수 항목 추출
                required_items = bullet_pattern.findall(section_content)
                if required_items:
                    structured_sections[section_name] = required_items
        
        return structured_sections

    def _build_full_conversation_text(self, conversation_log: list) -> str:
        """전체 대화를 하나의 텍스트로 변환"""
        
        chunks = []
        current_chunk = []
        max_chunk_size = 20  # 최대 청크 크기
        
        i = 0
        while i < len(conversation_log):
            msg = conversation_log[i]
            current_chunk.append(msg)
            role = msg.get("role", "")
            
            # 청크 크기가 최대치에 도달했을 때
            if len(current_chunk) >= max_chunk_size:
                # 현재 메시지가 의사 질문이면 환자 답변까지 기다림
                if role == "student":
                    # 다음 환자 답변을 찾아서 포함시킴
                    while i + 1 < len(conversation_log):
                        i += 1
                        next_msg = conversation_log[i]
                        current_chunk.append(next_msg)
                        if next_msg.get("role") == "patient":
                            break
                
                # 현재가 환자 답변이면, 다음이 또 환자 답변인지 확인
                elif role == "patient":
                    # 연속된 환자 답변들을 모두 포함
                    while i + 1 < len(conversation_log):
                        next_msg = conversation_log[i + 1]
                        if next_msg.get("role") == "patient":
                            i += 1
                            current_chunk.append(next_msg)
                        else:
                            break
                
                # 청크를 텍스트로 변환해서 저장
                chunk_text = ""
                for msg in current_chunk:
                    content = msg.get("content") or msg.get("text", "")
                    role = msg.get("role") or msg.get("speaker_role", "")
                    
                    if not content:
                        raise ValueError(f"메시지에 content가 없습니다: {msg}")
                    
                    if not role:
                        raise ValueError(f"메시지에 role이 없습니다: {msg}")
                        
                    if role == "student":
                        speaker = "의사"
                    elif role == "patient":
                        speaker = "환자"
                    else:
                        raise ValueError(f"알 수 없는 role입니다: {role}. 허용되는 role: student, patient")
                        
                    chunk_text += f"{speaker}: {content}\n"
                
                chunks.append(chunk_text)
                current_chunk = []
            
            i += 1
        
        # 남은 메시지들 처리
        if current_chunk:
            chunk_text = ""
            for msg in current_chunk:
                content = msg.get("content") or msg.get("text", "")
                role = msg.get("role") or msg.get("speaker_role", "")
                
                if not content:
                    raise ValueError(f"메시지에 content가 없습니다: {msg}")
                
                if not role:
                    raise ValueError(f"메시지에 role이 없습니다: {msg}")
                    
                if role == "student":
                    speaker = "의사"
                elif role == "patient":
                    speaker = "환자"
                else:
                    raise ValueError(f"알 수 없는 role입니다: {role}. 허용되는 role: student, patient")
                    
                chunk_text += f"{speaker}: {content}\n"
            
            chunks.append(chunk_text)
        
        return chunks

    def check_all_guidelines_in_chunk(self, chunk_text: str, structured_sections: dict, area_name: str) -> dict:
        """청크에서 모든 가이드라인 섹션을 한번에 체크 (효율적)"""
        
        if not chunk_text.strip():
            return {section_name: {"found_evidence": False} for section_name in structured_sections.keys()}
        
        # 모든 섹션을 하나의 프롬프트로 체크
        sections_text = ""
        for section_name, questions in structured_sections.items():
            sections_text += f"\n【{section_name}】\n"
            for q in questions:
                sections_text += f"  • {q}\n"
        
        # 고도화된 평가 프롬프트
        prompt = f"""당신은 CPX(Clinical Performance Examination) 평가 전문가입니다.
의과대학생이 표준화 환자와 나눈 대화를 분석하여 각 가이드라인 항목이 적절히 다뤄졌는지 평가하세요.

=== 평가 대상 대화 ===
{chunk_text}

=== {area_name} 평가 가이드라인 ===
{sections_text}

=== 영역별 평가 원칙 ===
{self._get_area_specific_guidelines(area_name)}

## 평가 방법
1. **각 가이드라인 항목별로** 해당 항목의 required_questions/actions와 **의미적으로 유사한** 의사 질문이 있는지 확인
2. **1순위**: 의사가 해당 항목의 required_questions와 비슷한 질문을 했으면 "found: true"
3. **2순위**: 의사 질문이 전혀 없지만 환자가 관련 정보를 자발적으로 언급했으면 "found: true"
4. **완전히 다른 주제의 질문이거나 전혀 언급되지 않았으면** "found: false"

## Evidence 수집 규칙
 **해당 항목의 required_questions/actions와 의미적으로 일치하는 질문만 evidence로 사용**
- **1순위**: 해당 항목의 required_questions와 비슷한 의미의 "의사:" 질문 문장
- **2순위**: 의사가 해당 항목에 대해 전혀 질문하지 않았지만 환자가 관련 정보를 언급한 "환자:" 문장
- **주의**: 완전히 다른 주제의 의사 질문은 evidence가 될 수 없음
- **예시**: "과거력" 항목에서 "의사: 가족 중에 치매 환자가 있으세요?" → 이건 가족력 질문이므로 과거력 evidence가 될 수 없음

## 유연한 판단 기준
- 표현이 달라도 의미가 같으면 인정
- 직접적 질문이 아니어도 관련 정보를 얻으려는 의도가 명확하면 인정
- 한 번의 질문으로 여러 항목을 커버할 수 있음

각 항목별로 정확히 평가하여 답변:
{{
{', '.join([f'    "{section_name}": {{"found": true/false, "evidence": "해당 항목의 required_questions와 일치하는 의사 질문 또는 빈 문자열"}}' for section_name in structured_sections.keys()])}
}}"""
        
        print("=" * 80)
        print("📋 LLM 프롬프트:")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.invoke(messages)
            result_text = response.content
            
            # JSON 추출
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # 기존 형식으로 변환
                converted_result = {}
                for section_name in structured_sections.keys():
                    section_data = result.get(section_name, {})
                    converted_result[section_name] = {
                        "found_evidence": section_data.get("found", False),
                        "evidence": section_data.get("evidence", ""),
                        "answered_questions": [section_name] if section_data.get("found", False) else []
                    }
                
                return converted_result
            else:
                print(f"❌ JSON 형식을 찾을 수 없습니다: {result_text[:100]}")
                return {section_name: {"found_evidence": False} for section_name in structured_sections.keys()}
                
        except Exception as e:
            print(f"❌ 청크 분석 실패: {e}")
            return {section_name: {"found_evidence": False} for section_name in structured_sections.keys()}

    def _get_area_specific_guidelines(self, area_name: str) -> str:
        """영역별 특화된 평가 원칙 반환"""
        
        if area_name == "병력 청취":
            return """
### 병력 청취 평가 원칙
- 각 섹션의 **핵심 주제**에 대해 의사가 적절히 **질문**했는지 평가
- **모든 개별 질문을 다 물어볼 필요 없음** - 비슷한 질문들은 하나로 커버 가능
- **환자로부터 해당 정보를 얻었다면 완료로 인정** (질문 방식이나 표현 무관)
- **관련된 주제를 다뤘다면 해당 섹션으로 인정** (간접적 질문도 포함)
- 환자가 먼저 정보를 제공해도 의사가 확인했다면 완료로 인정

### 섹션별 유연한 해석 기준
- **각 섹션의 가이드라인 질문과 비슷한 내용을 다뤘는지 판단**
- **표현이 달라도 같은 정보를 얻으려는 의도면 해당 섹션으로 인정**
- **환자 응답을 통해 해당 섹션의 목적에 맞는 정보를 확인했다면 완료**
- **직접적 질문이 아니어도 관련 정보를 얻었다면 인정**"""
            
        elif area_name == "신체 진찰":
            return """
### 신체 진찰 평가 원칙
- **가이드라인에 명시된 검사명을 언급했는지 확인**
- **"○○ 검사를 시행하겠습니다" 형태로 언급하면 완료로 인정**
- **현재 시점에서 실시하는 검사와 미래 계획은 구분**
- **환자교육에서 "나중에 필요한 검사" 언급은 신체진찰과 별개**

### 완료 조건 (현실적)
- 【진찰 준비】: 환자에게 진찰 시작 안내
- 【검사 수행】: 가이드라인의 구체적 검사명 언급 (질환별로 다름)
- **"지금 ○○검사를 하겠습니다" = 완료**, **"추후 ○○검사가 필요합니다" = 환자교육**"""
            
        elif area_name == "환자 교육":
            return """
### 환자 교육 평가 원칙
- 의사가 환자에게 **설명, 안내, 교육**을 제공했는지 평가
- **공감**: 환자의 걱정이나 감정에 대한 이해 표현
- **추정 진단**: 구체적 질환명을 언급한 진단 제시
- **검사 계획**: 구체적 검사명이나 검사 방법 언급
- **치료 계획**: 향후 치료 방향이나 관리 방법 안내

### 구체적 예시
- 【공감】: "많이 걱정되셨을 것 같습니다" → 완료
- 【추정 진단】: "○○질환 가능성을 고려해야 할 것 같습니다" → 완료
- 【검사 계획】: "○○ 검사를 통해 확인하겠습니다" → 완료
- 【치료 계획】: "○○ 치료를 시작하고 정기적으로 경과를 보겠습니다" → 완료"""
            
        else:
            return """
### 일반 평가 원칙
- 각 섹션의 핵심 주제가 적절히 다뤄졌는지 평가
- 표현이 달라도 의미상 동일한 내용을 확인했다면 완료로 판단"""

    def evaluate_area_with_chunks(self, conversation_log: list, area_name: str, structured_sections: dict) -> dict:
        """청크 기반으로 영역 평가 - 즉시 체크 방식"""
        
        # 1. 대화를 청크로 분할하고 텍스트로 변환
        chunk_texts = self.split_and_build_chunks(conversation_log)
        
        # 2. 가이드라인 항목별 체크리스트 초기화
        guideline_checklist = {}
        for section_name, questions in structured_sections.items():
            guideline_checklist[section_name] = {
                "completed": False,
                "evidence": ""
            }
        
        # 3. 각 청크에서 모든 가이드라인을 한번에 체크
        for i, chunk_text in enumerate(chunk_texts):
            chunk_result = self.check_all_guidelines_in_chunk(chunk_text, structured_sections, area_name)
            
            # 결과를 각 가이드라인 체크리스트에 반영
            for section_name, section_result in chunk_result.items():
                if section_name in guideline_checklist and not guideline_checklist[section_name]["completed"]:
                    checklist = guideline_checklist[section_name]
                    
                    if section_result.get("found_evidence"):
                        checklist["completed"] = True
                        if not checklist["evidence"]:  # 첫 번째 evidence만 저장
                            evidence = section_result.get("evidence", "")
                            if evidence:
                                checklist["evidence"] = evidence
        
        # 4. 단순한 결과 생성
        guideline_evaluations = []
        completed_count = 0
        
        for section_name, checklist in guideline_checklist.items():
            if checklist["completed"]:
                completed_count += 1
            
            guideline_evaluations.append({
                "guideline_item": section_name,
                "completed": checklist["completed"],
                "evidence": checklist["evidence"]
            })
        
        # 5. 완성도 계산
        total_count = len(guideline_checklist)
        completion_rate = completed_count / total_count if total_count > 0 else 0
        

        
        return {
            "area_name": area_name,
            "completion_rate": f"{completion_rate:.1%}",
            "guideline_evaluations": guideline_evaluations,
            "total_guidelines": total_count,
            "completed_guidelines": completed_count
        }

    async def start_evaluation_session(self, user_id: str, scenario_id: str, result_id: Optional[int] = None) -> str:
        """평가 세션 시작"""
        session_id = f"{user_id}_{scenario_id}_{int(datetime.now().timestamp())}"
        
        self.session_data[session_id] = {
            "user_id": user_id,
            "scenario_id": scenario_id,
            "result_id": result_id,  # CPX result_id 저장
            "start_time": datetime.now(),
            "conversation_entries": [],  # 실시간 대화 데이터
            # "audio_files": [],  # 임시 저장된 wav 파일 경로들
            "status": "active"
        }
        
        return session_id

    async def add_conversation_entry(self, session_id: str, audio_file_path: str, 
                                   text: str, speaker_role: str, emotion_analysis: Optional[Dict] = None) -> Dict:
        """실시간 대화 엔트리 추가 (SER 결과는 queue에서 전달받음)"""
        if session_id not in self.session_data:
            return {"error": "세션을 찾을 수 없습니다"}
        
        try:
            timestamp = datetime.now()
            
            # SER 결과 로깅 (queue에서 전달받은 경우)
            if emotion_analysis:
                print(f"🎭 [{session_id}] 감정 분석 결과 수신: {emotion_analysis['predicted_emotion']} ({emotion_analysis['confidence']:.2f})")
            
            # 대화 엔트리 생성
            conversation_entry = {
                "timestamp": timestamp.isoformat(),
                "text": text,
                "emotion": emotion_analysis,
                "speaker_role": speaker_role,  # "student" (의사) or "patient" (환자)
                "audio_file_path": audio_file_path
            }
            
            # 세션 데이터에 추가
            session = self.session_data[session_id]
            session["conversation_entries"].append(conversation_entry)
            if "audio_files" not in session:
                session["audio_files"] = []
            session["audio_files"].append(audio_file_path)
            
            print(f"📝 [{session_id}] 대화 엔트리 추가: {speaker_role} - {text[:50]}...")
            
            # 평가 완료 후 임시 WAV 파일들 삭제
            try:
                await self._cleanup_audio_files(audio_file_path)
            except Exception as e:
                print(f"❌ [{audio_file_path}] 임시 WAV 파일 삭제 실패: {e}")
            
            return {
                "success": True,
                "entry": conversation_entry,
                "total_entries": len(session["conversation_entries"])
            }
            
        except Exception as e:
            print(f"❌ [{session_id}] 대화 엔트리 추가 실패: {e}")
            return {"error": str(e)}

    async def end_evaluation_session(self, session_id: str) -> Dict:
        """평가 세션 종료 및 종합 평가 실행"""
        if session_id not in self.session_data:
            return {"error": "세션을 찾을 수 없습니다"}
        
        session = self.session_data[session_id]
        session["end_time"] = datetime.now()
        session["status"] = "completed"
        
        # 종합 평가 실행
        evaluation_result = await self._comprehensive_evaluation(session_id, session)
        
        # CPX 데이터베이스 업데이트
        await self._update_cpx_database_after_evaluation(session_id, evaluation_result)
        
        return evaluation_result

    def get_session_summary(self, user_id: str) -> list:
        """사용자의 세션 요약"""
        return [
            {
                "session_id": sid,
                "scenario_id": data["scenario_id"],
                "date": data["start_time"].strftime("%Y-%m-%d %H:%M"),
                "status": data["status"]
            }
            for sid, data in self.session_data.items()
            if data["user_id"] == user_id
        ]

    async def _comprehensive_evaluation(self, session_id: str, session: Dict) -> Dict:
        """종합적인 세션 평가 수행 (SER + LangGraph 통합)"""
        print(f"🔍 [{session_id}] 종합 평가 시작...")
        
        # LangGraph 기반 텍스트 평가 (새로운 대화 데이터 사용)
        langgraph_analysis = None
        if self.llm and self.workflow:
            try:
                # 새로운 conversation_entries를 conversation_log 형식으로 변환
                conversation_log = []
                for entry in session.get("conversation_entries", []):
                    conversation_log.append({
                        "role": entry["speaker_role"],
                        "content": entry["text"],
                        "timestamp": entry["timestamp"],
                        "emotion": entry.get("emotion")
                    })
                
                if conversation_log:  # 대화 데이터가 있는 경우에만 평가
                    langgraph_analysis = await self.evaluate_conversation_with_langgraph(
                        session["user_id"], 
                        session["scenario_id"], 
                        conversation_log
                    )
                    print(f"✅ [{session_id}] LangGraph 텍스트 평가 완료")
                else:
                    print(f"⚠️ [{session_id}] 대화 데이터가 없어 LangGraph 평가를 건너뜁니다")
                
            except Exception as e:
                print(f"❌ [{session_id}] LangGraph 텍스트 평가 실패: {e}")
                langgraph_analysis = {"error": str(e)}
        
        # 종합 결과 구성
        evaluation_result = {
            "session_id": session_id,
            "user_id": session["user_id"],
            "scenario_id": session["scenario_id"],
            "start_time": session["start_time"].isoformat(),
            "end_time": session["end_time"].isoformat(),
            "duration_minutes": (session["end_time"] - session["start_time"]).total_seconds() / 60,
            
            # 상세 분석 결과
            "langgraph_text_analysis": langgraph_analysis,  # LangGraph 기반 텍스트 평가 결과
            
            # 실시간 대화 데이터 (감정 분석 포함)
            "conversation_entries": [
                {
                    "timestamp": entry["timestamp"],
                    "text": entry["text"],
                    "speaker_role": entry["speaker_role"],
                    "emotion": entry.get("emotion"),
                    "audio_file": entry["audio_file_path"]
                }
                for entry in session.get("conversation_entries", [])
            ]
        }
        
        print(f"✅ [{session_id}] 종합 평가 완료")
        return evaluation_result

    async def _save_evaluation_result(self, session_id: str, result: Dict):
        """평가 결과를 파일로 저장"""
        try:
            # JSON 파일로 저장
            json_path = self.evaluation_dir / f"{session_id}_evaluation.json"
            
            async with aiofiles.open(json_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(result, ensure_ascii=False, indent=2))
            
            print(f"💾 [{session_id}] 평가 결과 저장 완료: {json_path}")
            
        except Exception as e:
            print(f"❌ [{session_id}] 평가 결과 저장 실패: {e}")

    async def get_evaluation_result(self, session_id: str) -> Dict:
        """저장된 평가 결과 조회"""
        json_path = self.evaluation_dir / f"{session_id}_evaluation.json"
        
        if not json_path.exists():
            return {"error": "평가 결과를 찾을 수 없습니다"}
        
        try:
            async with aiofiles.open(json_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            return {"error": f"평가 결과 로드 실패: {e}"}

    # LangGraph 기반 텍스트 평가 기능
    
    def _initialize_langgraph_components(self):
        """LangGraph 컴포넌트들 초기화"""
        try:
            # OpenAI API 설정
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm = ChatOpenAI(
                    openai_api_key=api_key,
                    model_name="gpt-3.5-turbo",
                    temperature=0.3,
                    max_tokens=4000
                )
                
                # 워크플로우 생성
                self.workflow = self._create_evaluation_workflow()
                print("✅ LangGraph 텍스트 평가 컴포넌트 초기화 완료")
            else:
                print("⚠️ OPENAI_API_KEY가 설정되지 않아 텍스트 평가 기능을 사용할 수 없습니다")
                
        except Exception as e:
            print(f"❌ LangGraph 컴포넌트 초기화 실패: {e}")
            self.llm = None
            self.workflow = None



    async def evaluate_conversation(self, user_id: str, scenario_id: str, conversation_log: List[Dict]) -> Dict:
        """LangGraph 워크플로우를 사용한 CPX 평가 실행"""
        # 초기 상태 구성 (Multi-Step 전용)
        initial_state = CPXEvaluationState(
            user_id=user_id,
            scenario_id=scenario_id,
            conversation_log=conversation_log,
            medical_context_analysis=None,
            question_intent_analysis=None,
            completeness_assessment=None,
            quality_evaluation=None,
            appropriateness_validation=None,
            comprehensive_evaluation=None,
            final_scores=None,
            feedback=None,
            evaluation_metadata=None,
            messages=[]
        )
        
        try:
            # 워크플로우 실행
            print(f"🚀 [{user_id}] CPX 평가 워크플로우 시작")
            final_state = self.workflow.invoke(initial_state)
            
            # 간단한 대화 요약 정보 생성
            student_questions = [msg for msg in conversation_log if msg.get("role") == "student"]
            conversation_summary = {
                "total_questions": len(student_questions),
                "duration_minutes": len(conversation_log) * 0.5
            }
            
            # 최종 결과 구성
            result = {
                "evaluation_metadata": final_state.get("evaluation_metadata", {}),
                "scores": final_state.get("final_scores", {}),
                "feedback": final_state.get("feedback", {}),
                "conversation_summary": conversation_summary,
                "detailed_analysis": {
                    "medical_context": final_state.get("medical_context_analysis", {}),
                    "question_intent": final_state.get("question_intent_analysis", {}),
                    "completeness": final_state.get("completeness_assessment", {}),
                    "quality": final_state.get("quality_evaluation", {}),
                    "appropriateness": final_state.get("appropriateness_validation", {}),
                    "comprehensive": final_state.get("comprehensive_evaluation", {})
                },
                "evaluation_method": "6단계 의학적 분석",
                "system_info": {
                    "version": "v2.0",
                    "evaluation_steps": 6
                }
            }
            
            print(f"🎉 [{user_id}] CPX 평가 워크플로우 완료")
            
            return result
            
        except Exception as e:
            print(f"❌ [{user_id}] 평가 워크플로우 오류: {e}")
            return {
                "error": str(e),
                "user_id": user_id,
                "scenario_id": scenario_id,
                "evaluation_date": datetime.now().isoformat()
            }

    def _create_evaluation_workflow(self):
        """CPX 평가 워크플로우 생성 (3단계 명확화)"""
        workflow = StateGraph(CPXEvaluationState)

        workflow.add_node("initialize", self._initialize_evaluation)
        workflow.add_node("step1_rag_completeness", self._evaluate_rag_completeness)
        workflow.add_node("step2_quality_assessment", self._evaluate_quality_assessment)
        workflow.add_node("step3_comprehensive_results", self._generate_comprehensive_results)

        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "step1_rag_completeness")
        workflow.add_edge("step1_rag_completeness", "step2_quality_assessment")
        workflow.add_edge("step2_quality_assessment", "step3_comprehensive_results")
        workflow.add_edge("step3_comprehensive_results", END)

        return workflow.compile()

    def _initialize_evaluation(self, state: CPXEvaluationState) -> CPXEvaluationState:
        print(f"🎯 [{state['user_id']}] CPX 평가 초기화 - 시나리오: {state['scenario_id']}")
        
        metadata = {
            "user_id": state["user_id"],
            "scenario_id": state["scenario_id"],
            "evaluation_date": datetime.now().isoformat(),
            "conversation_duration_minutes": len(state["conversation_log"]) * 0.5,
            "voice_recording_path": "s3로 저장",
            "conversation_transcript": json.dumps(state["conversation_log"], ensure_ascii=False)
        }
        
        return {
            **state,
            "evaluation_metadata": metadata,
            "messages": [HumanMessage(content="CPX 평가를 시작합니다.")]
        }

    def _evaluate_rag_completeness(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """1단계: RAG 기반 완성도 평가 (병력청취, 신체진찰, 환자교육)"""
        print(f"📋 [{state['user_id']}] 1단계: RAG 기반 완성도 평가 시작")
        
        conversation_text = self._build_conversation_text(state["conversation_log"])
        scenario_id = state["scenario_id"]
        
        # 시나리오에서 카테고리 정보 로드
        scenario_category = self._get_scenario_category(scenario_id)
        if not scenario_category:
            raise ValueError(f"시나리오 '{scenario_id}'의 카테고리를 찾을 수 없습니다.")
            
        rag_data = {
            "scenario_id": scenario_id,
            "category": scenario_category
        }
        
        # 3개 영역별 청크 기반 평가
        areas_evaluation = {}
        
        for area_key, area_name in [("history_taking", "병력 청취"), ("physical_examination", "신체 진찰"), ("patient_education", "환자 교육")]:
            # RAG에서 가이드라인 가져오기
            criteria_data = self.guideline_retriever.get_evaluation_criteria(scenario_category, area_name)
            documents = criteria_data.get("documents", [])
            
            if not documents or not documents[0]:
                raise ValueError(f"❌ {area_name} 가이드라인을 찾을 수 없습니다.")
            
            # 구조화된 섹션 파싱
            structured_sections = self._parse_structured_sections(documents[0])
            
            # 청크 기반 평가 실행
            areas_evaluation[area_key] = self.evaluate_area_with_chunks(
                state["conversation_log"], area_name, structured_sections
            )
        
        # 전체 완성도 점수 계산
        total_guidelines = sum(area.get("total_guidelines", 0) for area in areas_evaluation.values())
        completed_guidelines = sum(area.get("completed_guidelines", 0) for area in areas_evaluation.values())
        overall_completeness = completed_guidelines / total_guidelines if total_guidelines > 0 else 0
        
        # 전체 완료/누락 항목 수집 (새로운 JSON 형식 대응)
        all_completed_items = []
        all_missing_items = []
        for area_data in areas_evaluation.values():
            # 기존 형식 지원 (하위 호환성)
            if "completed_items" in area_data:
                all_completed_items.extend(area_data.get("completed_items", []))
            if "missing_items" in area_data:
                all_missing_items.extend(area_data.get("missing_items", []))
            
            # 새로운 section_evaluations 형식 지원
            section_evals = area_data.get("section_evaluations", [])
            for section in section_evals:
                if section.get("status") == "completed":
                    covered_desc = section.get("how_covered", "")
                    if covered_desc:
                        all_completed_items.append(f"{section.get('section_name', '')}: {covered_desc}")
                elif section.get("status") in ["partial", "missing"]:
                    missing_aspects = section.get("missing_aspects", [])
                    if missing_aspects:
                        all_missing_items.extend([f"{section.get('section_name', '')}: {aspect}" for aspect in missing_aspects])
                    elif section.get("status") == "missing":
                        all_missing_items.append(f"{section.get('section_name', '')}: 전체 누락")
        
        rag_completeness_result = {
            "category": scenario_category or scenario_id,
            "overall_completeness": round(overall_completeness, 2),
            "areas_evaluation": areas_evaluation,
            "total_completed_items": len(all_completed_items),
            "total_missing_items": len(all_missing_items),
            "completed_items": all_completed_items,
            "missing_items": all_missing_items,
            "evaluation_method": "rag_three_areas"
        }
        
        print(f"✅ [{state['user_id']}] 1단계: RAG 기반 완성도 평가 완료 - 완성도: {overall_completeness:.2%}")
        
        return {
            **state,
            "completeness_assessment": rag_completeness_result,
            "messages": state["messages"] + [HumanMessage(content="1단계: RAG 기반 완성도 평가 완료")]
        }

    


    def _build_conversation_text(self, conversation_log: List[Dict]) -> str:
        """대화 로그를 텍스트로 변환"""
        conversation_parts = []
        for msg in conversation_log:
            speaker = "학생" if msg.get("role") == "student" else "환자"
            content = msg.get("content", "")
            conversation_parts.append(f"{speaker}: {content}")
        return "\n".join(conversation_parts)

    def _evaluate_quality_assessment(self, state: CPXEvaluationState) -> CPXEvaluationState:
        print(f"⭐ [{state['user_id']}] 2단계: 품질 평가 시작")
        
        conversation_text = self._build_conversation_text(state["conversation_log"])
        scenario_id = state["scenario_id"]
        scenario_info = self.scenario_applicable_elements.get(scenario_id, {})
        
        quality_assessment_prompt = f"""
당신은 의학교육 평가 전문가입니다. 다음 CPX 대화의 품질을 4가지 기준으로 평가하세요.

【학생-환자 대화】: {conversation_text}
【시나리오 정보】: {scenario_info.get('name', scenario_id)}

다음 4가지 품질 기준으로 평가하세요:

【1. 의학적 정확성 (Medical Accuracy)】:
- 질문의 의학적 타당성과 정확성
- 진단적 접근의 논리성
- 의학 용어 사용의 적절성
- 임상적 판단의 합리성

【2. 의사소통 효율성 (Communication Efficiency)】:
- 환자가 이해하기 쉬운 언어 사용
- 질문의 명확성과 구체성
- 환자 반응에 대한 적절한 후속 질문
- 대화 흐름의 자연스러움

【3. 전문성 (Professionalism)】:
- 의료진다운 태도와 예의
- 환자에 대한 공감과 배려
- 체계적이고 논리적인 접근
- 자신감 있는 진료 태도

【4. 시나리오 적합성 (Scenario Appropriateness)】:
- 주어진 시나리오에 맞는 접근
- 환자 연령/성별/상황 고려
- 시간 제약 내 효율적 진행
- 우선순위에 따른 체계적 접근

각 항목을 1-10점으로 평가하고, 전체 품질 점수를 산출하세요.

다음 JSON 형식으로 응답하세요:
{{
    "medical_accuracy": 의학적정확성점수(1-10),
    "communication_efficiency": 의사소통효율성점수(1-10),
    "professionalism": 전문성점수(1-10),
    "scenario_appropriateness": 시나리오적합성점수(1-10),
    "overall_quality_score": 전체품질점수(1-10),
    "quality_strengths": ["품질 면에서 우수한 점들"],
    "quality_improvements": ["품질 면에서 개선이 필요한 점들"],
    "detailed_analysis": {{
        "medical_accuracy_detail": "의학적 정확성에 대한 구체적 분석",
        "communication_detail": "의사소통에 대한 구체적 분석",
        "professionalism_detail": "전문성에 대한 구체적 분석",
        "scenario_fit_detail": "시나리오 적합성에 대한 구체적 분석"
    }}
}}
"""
        
        try:
            messages = [
                SystemMessage(content="당신은 의학교육 평가 전문가입니다."),
                HumanMessage(content=quality_assessment_prompt)
            ]
            
            response = self.llm.invoke(messages)
            # response가 dict인 경우와 객체인 경우 모두 처리
            if hasattr(response, 'content'):
                result_text = response.content.strip()
            elif isinstance(response, dict) and 'content' in response:
                result_text = response['content'].strip()
            else:
                result_text = str(response).strip()
            
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                quality_assessment = json.loads(json_match.group())
                
                print(f"✅ [{state['user_id']}] 2단계: 품질 평가 완료 - 종합 점수: {quality_assessment.get('overall_quality_score', 0)}")
                
                return {
                    **state,
                    "quality_evaluation": quality_assessment,
                    "messages": state["messages"] + [HumanMessage(content="2단계: 품질 평가 완료")]
                }
            else:
                raise ValueError(f"품질 평가에서 JSON 패턴을 찾을 수 없습니다. LLM 응답: {result_text}")
        except Exception as e:
            print(f"❌ [{state['user_id']}] 품질 평가 실패: {e}")
            raise e

    def _generate_comprehensive_results(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """3단계: 종합 평가 및 최종 결과 생성"""
        print(f"🎯 [{state['user_id']}] 3단계: 종합 평가 시작")
        
        # 1단계와 2단계 결과 수집
        rag_completeness = state.get("completeness_assessment", {})
        quality_assessment = state.get("quality_evaluation", {})
        
        # 최종 점수 계산 (가중치: 완성도 60%, 품질 40%)
        completeness_score = rag_completeness.get("overall_completeness", 0.5) * 10  # 0-10 스케일로 변환
        quality_score = quality_assessment.get("overall_quality_score", 6)
        
        # 가중치 적용: 완성도 60%, 품질 40%
        final_score = (completeness_score * 0.6) + (quality_score * 0.4)
        final_score = min(10, max(0, final_score))  # 0-10 범위로 제한
        
        # 종합 피드백 생성
        strengths = []
        improvements = []
        
        # 1단계 RAG 평가에서 강점/개선점 수집
        for area_name, area_data in rag_completeness.get("areas_evaluation", {}).items():
            if isinstance(area_data, dict):
                strengths.extend(area_data.get("strengths", []))
                improvements.extend(area_data.get("improvements", []))
        
        # 2단계 품질 평가에서 강점/개선점 추가
        strengths.extend(quality_assessment.get("quality_strengths", []))
        improvements.extend(quality_assessment.get("quality_improvements", []))
        
        # 상세 분석 생성
        detailed_analysis_parts = []
        detailed_analysis_parts.append(f"【완성도 평가】 RAG 기반 평가 결과 {rag_completeness.get('overall_completeness', 0):.1%} 완성")
        detailed_analysis_parts.append(f"【품질 평가】 4가지 품질 기준 평균 {quality_score:.1f}점")
        
        if rag_completeness.get("total_completed_items", 0) > 0:
            detailed_analysis_parts.append(f"총 {rag_completeness.get('total_completed_items', 0)}개 항목 완료")
        
        if rag_completeness.get("total_missing_items", 0) > 0:
            detailed_analysis_parts.append(f"{rag_completeness.get('total_missing_items', 0)}개 항목 누락")
        
        comprehensive_result = {
            "final_score": round(final_score, 1),
            "grade": self._calculate_grade(final_score * 10),  # 100점 스케일로 변환
            "score_breakdown": {
                "completeness_score": round(completeness_score, 1),
            "quality_score": round(quality_score, 1),
                "weighted_completeness": round(completeness_score * 0.6, 1),
                "weighted_quality": round(quality_score * 0.4, 1)
            },
            "detailed_feedback": {
                "strengths": list(set(strengths))[:5] if strengths else ["평가를 성실히 완료했습니다"],
                "improvements": list(set(improvements))[:5] if improvements else ["지속적인 학습과 연습이 필요합니다"],
                "overall_analysis": " | ".join(detailed_analysis_parts)
            },
            "evaluation_summary": {
                "method": "3단계 RAG+품질 평가",
                "steps_completed": 3,
                "completeness_rate": rag_completeness.get("overall_completeness", 0),
                "quality_details": quality_assessment.get("detailed_analysis", {}),
                "total_items_evaluated": rag_completeness.get("total_completed_items", 0) + rag_completeness.get("total_missing_items", 0)
            }
        }
        
        print(f"✅ [{state['user_id']}] 3단계: 종합 평가 완료 - 최종 점수: {final_score:.1f}/10 ({comprehensive_result['grade']})")
        
        return {
            **state,
            "comprehensive_evaluation": comprehensive_result,
            "final_scores": {
                "total_score": round(final_score * 10, 1),  # 100점 스케일
                "completion_rate": rag_completeness.get("overall_completeness", 0.5),
                "quality_score": quality_score,
                "grade": comprehensive_result["grade"]
            },
            "feedback": comprehensive_result["detailed_feedback"],
            "messages": state["messages"] + [HumanMessage(content=f"3단계: 종합 평가 완료 - {final_score:.1f}점 ({comprehensive_result['grade']})")]
        }

    def _calculate_grade(self, score: float) -> str:
        """점수에 따른 등급 계산"""
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "B+"
        elif score >= 75:
            return "B"
        elif score >= 70:
            return "C+"
        elif score >= 65:
            return "C"
        else:
            return "F"

    async def _update_cpx_database_after_evaluation(self, session_id: str, evaluation_result: dict):
        """평가 완료 후 CPX Details만 업데이트"""
        try:
            session = self.session_data[session_id]
            result_id = session["result_id"]
            user_id = session["user_id"]
            
            # CPX Details만 업데이트 (시스템 평가 데이터)
            async for db in get_db():
                cpx_service = CpxService(db)
                
                await cpx_service.update_cpx_details(
                    result_id=result_id,
                    user_id=int(user_id),
                    system_evaluation_data=evaluation_result
                )
                
                print(f"✅ CPX Details 업데이트 완료: result_id={result_id}, session_id={session_id}")
                break
                
        except Exception as e:
            print(f"❌ CPX Details 업데이트 실패: {e}")

    async def _cleanup_audio_files(self, audio_file_path: str):
        """평가 완료 후 임시 WAV 파일들만 삭제 (TTS 캐시 파일은 보존)"""

        try:
            file_path_obj = Path(audio_file_path)
            # TTS 캐시 파일은 삭제하지 않음
            if "cache/tts" in str(file_path_obj):
                print(f"🔒 TTS 캐시 파일 보존: {audio_file_path}")
                return
                
            if file_path_obj.exists() and file_path_obj.suffix == '.wav':
                file_path_obj.unlink()  # WAV 파일만 삭제
                print(f"🗑️ 임시 WAV 파일 삭제: {audio_file_path}")
                    
        except Exception as e:
            print(f"❌ WAV 파일 삭제 실패 ({audio_file_path}): {e}")