"""
범용 의료 추출기 (간단 버전)
- 어떤 질병/증상이든 추출 가능
- BeautifulSoup 없이 LLM이 직접 HTML 처리
- 표준화된 4개 섹션 구조
"""

import json
import os
from typing import Dict, Optional, List

class MedicalExtractor:
    def __init__(self):
        """범용 의료 추출기 초기화"""
        # OpenAI API 키 확인
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1
        )
        
        self.system_prompt = """
당신은 의료 교재에서 CPX 체크리스트를 생성하는 전문가입니다.

주요 역할:
1. HTML에서 지정된 질병/증상 관련 모든 내용 식별
2. CPX 평가에 필요한 체크리스트 형태로 구조화:
   - 병력청취: OLDCART 패턴별 구체적 질문들
   - 신체진찰: 단계별 검사 항목들
   - 환자교육: 설명해야 할 내용들
3. 실제 CPX에서 바로 사용할 수 있는 실용적 형태
4. 각 항목별 구체적인 질문/행동 지침 제공

출력: CPX 체크리스트 JSON 형식
"""

    def extract_from_json_file(self, json_file_path: str, target_condition: str) -> Optional[Dict]:
        """
        JSON 파일에서 특정 질병/증상 내용 추출
        
        Args:
            json_file_path: JSON 청크 파일 경로
            target_condition: 추출할 질병/증상명 (예: "기억력 저하", "흉통")
            
        Returns:
            추출된 의료 가이드 또는 None
        """
        
        print(f"📄 파일 처리: {json_file_path}")
        print(f"🎯 대상 질병: {target_condition}")
        
        # 1. JSON에서 HTML 추출
        html_content = self._load_html_from_json(json_file_path)
        if not html_content:
            return None
        
        # 2. HTML 크기 제한
        if len(html_content) > 100000:
            html_content = html_content[:100000] + "\n[... 내용 생략 ...]"
            print(f"⚠️ HTML 크기 제한: {len(html_content):,}자")
        
        # 3. 추출 실행 (LLM이 직접 관련성 판단)
        result = self._extract_with_llm(html_content, target_condition)
        
        if result:
            return self._post_process(result, target_condition)
        
        return None

    def _load_html_from_json(self, json_file_path: str) -> Optional[str]:
        """JSON 파일에서 HTML 내용 추출"""
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "content" in data and "html" in data["content"]:
                html_content = data["content"]["html"]
                print(f"✅ HTML 추출: {len(html_content):,}자")
                return html_content
            else:
                print(f"❌ 예상 구조가 아닙니다. 키: {list(data.keys())}")
                return None
                
        except Exception as e:
            print(f"❌ JSON 로드 실패: {e}")
            return None

    def _extract_with_llm(self, html_content: str, target_condition: str) -> Optional[Dict]:
        """LLM을 사용해 의료 내용 추출"""
        
        extraction_prompt = f"""
다음 의료 교재 HTML에서 "{target_condition}" 관련 내용을 추출하여 CPX 체크리스트로 구조화해주세요:

대상 질병/증상: {target_condition}

HTML 내용:
{html_content}

완전한 CPX 체크리스트 구조로 출력해주세요. 다음 템플릿을 참고하되, HTML에서 찾은 실제 내용으로 구체적인 질문과 행동을 작성해주세요:

JSON 형식으로 출력:
{{
    "found": true/false,
    "category": "{target_condition}",
    "description": "{target_condition}을/를 주소로 내원한 환자에 대한 CPX 평가 체크리스트",
    "evaluation_areas": {{
        "history_taking": {{
            "name": "병력 청취",
            "subcategories": {{
                "O_onset": {{
                    "name": "O (Onset) - 발병 시기",
                    "required_questions": [
                        "언제부터 증상이 시작되었나요?",
                        "갑자기 시작되었나요? / 서서히 시작되었나요?",
                        "특별한 사건이 있지는 않았나요?"
                    ]
                }},
                "L_location": {{
                    "name": "L (Location) - 위치",
                    "applicable": false
                }},
                "D_duration": {{
                    "name": "D (Duration) - 지속시간/변동성",
                    "required_questions": [
                        "증상이 지속적인가요?",
                        "증상이 좋아지기도 하나요?"
                    ]
                }},
                "Co_course": {{
                    "name": "Co (Course) - 경과",
                    "required_questions": [
                        "증상이 점점 더 심해지시나요?"
                    ]
                }},
                "Ex_experience": {{
                    "name": "Ex (Experience) - 과거 경험",
                    "required_questions": [
                        "이전에도 이런 적이 있나요?",
                        "당시 치료를 받았나요?"
                    ]
                }},
                "C_character": {{
                    "name": "C (Character) - 증상 특징",
                    "required_questions": [
                        "HTML에서 찾은 해당 증상의 구체적 특징 질문들"
                    ]
                }},
                "A_associated": {{
                    "name": "A (Associated symptom) - 동반 증상 (감별진단별)",
                    "required_questions": [
                        "HTML에서 찾은 감별진단을 위한 동반 증상 질문들"
                    ]
                }},
                "F_factor": {{
                    "name": "F (Factor) - 악화/완화 요인",
                    "required_questions": [
                        "증상을 악화시키거나 완화시키는 요인들"
                    ]
                }},
                "E_exam": {{
                    "name": "E (Exam) - 이전 검사/건강검진",
                    "required_questions": [
                        "이전 건강검진에서 이상 소견은 없었나요?"
                    ]
                }},
                "trauma_history": {{
                    "name": "외상력",
                    "required_questions": [
                        "HTML에서 찾은 외상과 관련된 질문들"
                    ]
                }},
                "past_medical_history": {{
                    "name": "과거력",
                    "required_questions": [
                        "이전에 진단받은 질환이 있으신가요?",
                        "HTML에서 찾은 관련 진료 관련 질문들"
                    ]
                }},
                "medication_history": {{
                    "name": "약물력",
                    "required_questions": [
                        "현재 복용하시는 약물이 있나요?",
                        "HTML에서 찾은 약물 관련 구체적 질문들"
                    ]
                }},
                "social_history": {{
                    "name": "사회력",
                    "required_questions": [
                        "최종 학력은 어떻게 되시나요? 직업은 어떻게 되시나요?",
                        "술은 얼마나 드시나요? (빈도, 일회 섭취량)",
                        "흡연은 하시나요?"
                    ]
                }},
                "family_history": {{
                    "name": "가족력",
                    "required_questions": [
                        "HTML에서 찾은 가족력 관련 구체적 질문들"
                    ]
                }},
                "gynecologic_history": {{
                    "name": "여성력 (해당시)",
                    "required_questions": [
                        "LMP / 규칙적 / 주기 / 폐경"
                    ]
                }}
            }}
        }},
        "physical_examination": {{
            "name": "신체 진찰",
            "subcategories": {{
                "examination_preparation": {{
                    "name": "진찰 준비",
                    "required_actions": [
                        "진찰 시작 전 환자에게 설명하고 동의를 받기"
                    ]
                }},
                "examination": {{
                    "name": "검사",
                    "required_actions": [
                        "HTML에서 찾은 해당 증상에 관련된 구체적 검사 방법들"
                    ]
                }}
            }}
        }},
        "patient_education": {{
            "name": "환자 교육",
            "subcategories": {{
                "empathy": {{
                    "name": "공감",
                    "required_actions": [
                        "HTML에서 찾은 해당 증상에 적합한 공감 표현"
                    ]
                }},
                "suspected_diagnosis": {{
                    "name": "추정 진단",
                    "required_actions": [
                        "HTML에서 찾은 추정되는 진단에 대한 구체적 설명"
                    ]
                }},
                "differential_diagnosis": {{
                    "name": "감별 진단",
                    "required_actions": [
                        "HTML에서 찾은 다른 가능한 진단들에 대한 설명"
                    ]
                }},
                "diagnostic_tests": {{
                    "name": "검사 계획",
                    "required_actions": [
                        "HTML에서 찾은 필요한 검사들에 대한 구체적 설명"
                    ]
                }},
                "treatment_education": {{
                    "name": "치료 및 교육",
                    "required_actions": [
                        "HTML에서 찾은 치료 계획 및 생활 지도"
                    ]
                }},
                "final_questions": {{
                    "name": "마무리 질문",
                    "required_actions": [
                        "마지막으로 혹시 궁금한 점이 있으신가요?"
                    ]
                }}
            }}
        }}
    }},
    "keywords": ["HTML에서 추출된 관련 키워드들"],
    "confidence": 0.0-1.0
}}

중요한 지침:
1. 모든 질문과 행동은 HTML에서 찾은 실제 내용을 바탕으로 구체적으로 작성
2. 템플릿의 일반적인 내용을 HTML 내용으로 대체
3. L_location은 해당 증상에서 위치가 중요하지 않으면 applicable: false로 설정
4. 각 카테고리별로 HTML에서 찾은 실제 의료 내용을 반영
5. {target_condition}와 관련 없는 다른 질병 내용은 제외
"""
        
        try:
            print("🧠 LLM으로 내용 추출 중...")
            
            from langchain_core.messages import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=extraction_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # JSON 코드 블록 제거
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]  # ```json 제거
            if content.endswith("```"):
                content = content[:-3]  # ``` 제거
            content = content.strip()
            
            print(f"🔍 처리된 JSON 내용 (처음 200자): {content[:200]}")
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"❌ JSON 파싱 실패: {e}")
                print(f"   처리된 내용: {content[:500]}")
                return None
            
            if result.get("found"):
                print("✅ LLM 추출 성공!")
                return result
            else:
                print(f"❌ LLM이 '{target_condition}' 내용을 찾지 못했습니다.")
                return None
                
        except Exception as e:
            print(f"❌ LLM 추출 실패: {e}")
            return None

    def _post_process(self, llm_result: Dict, target_condition: str) -> Dict:
        """체크리스트 결과 후처리"""
        
        # 체크리스트 구조 그대로 사용
        checklist = {
            "id": f"{target_condition.replace(' ', '_').lower()}_guideline",
            "category": llm_result.get("category", target_condition),
            "description": llm_result.get("description", f"{target_condition} CPX 체크리스트"),
            "evaluation_areas": llm_result.get("evaluation_areas", {}),
            "metadata": {
                "condition": target_condition,
                "keywords": llm_result.get("keywords", []),
                "extraction_method": "llm_guideline",
                "confidence": llm_result.get("confidence", 0.5),
                "total_questions": self._count_questions(llm_result.get("evaluation_areas", {}))
            }
        }
        
        return checklist

    def _count_questions(self, evaluation_areas: Dict) -> int:
        """체크리스트 내 총 질문/행동 개수 계산"""
        
        total_count = 0
        
        for area in evaluation_areas.values():
            if isinstance(area, dict) and "subcategories" in area:
                for subcategory in area["subcategories"].values():
                    if isinstance(subcategory, dict):
                        # required_questions 또는 required_actions 개수 계산
                        questions = subcategory.get("required_questions", [])
                        actions = subcategory.get("required_actions", [])
                        total_count += len(questions) + len(actions)
        
        return total_count
