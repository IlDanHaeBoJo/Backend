from sqlalchemy import select
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession # AsyncSession 임포트
from core.models import CpxResults, CpxDetails, CpxEvaluations, User
from typing import List, Optional, Dict
import json

class CpxService:
    def __init__(self, db: AsyncSession): # AsyncSession으로 타입 힌트 변경
        self.db = db

    async def create_cpx_result(self, student_id: int, patient_name: str, evaluation_status: str = "진행중") -> CpxResults:
        """
        새로운 CPX 실습 결과를 생성하고, 연관된 CpxDetails 및 CpxEvaluations를 함께 생성합니다.
        """
        cpx_result = CpxResults(
            student_id=student_id,
            patient_name=patient_name,
            evaluation_status=evaluation_status
        )
        self.db.add(cpx_result)
        await self.db.flush() # await 추가

        cpx_detail = CpxDetails(
            result_id=cpx_result.result_id,
            memo="", # 초기값 설정
            system_evaluation_data={} # 초기값 설정
        )
        self.db.add(cpx_detail)

        # CpxEvaluations는 초기 상태로 생성 (아직 평가되지 않음)
        cpx_evaluation = CpxEvaluations(
            result_id=cpx_result.result_id,
            evaluator_id=student_id, # 초기 평가자는 학생 본인으로 설정하거나, null 허용 시 null로 설정
            overall_score=0, # 초기 점수
            detailed_feedback="아직 평가되지 않았습니다.", # 초기 피드백
            evaluation_status="평가대기"
        )
        self.db.add(cpx_evaluation)

        await self.db.commit() # await 추가
        await self.db.refresh(cpx_result) # await 추가
        await self.db.refresh(cpx_detail) # await 추가
        await self.db.refresh(cpx_evaluation) # await 추가
        return cpx_result

    async def get_cpx_results_for_user(self, user_id: int) -> List[CpxResults]:
        """
        특정 사용자의 CPX 실습 결과 목록을 반환합니다.
        이 함수는 CpxDetails나 CpxEvaluations를 로드하지 않고 CpxResults의 기본 정보만 반환합니다.
        """
        result = await self.db.execute(
            select(CpxResults).filter(CpxResults.student_id == user_id)
        )
        return result.scalars().all()

    async def get_cpx_details_with_evaluations(self, result_id: int, user_id: int) -> Optional[CpxResults]:
        """
        특정 result_id에 해당하는 CPX 실습 결과와 그에 연결된 CpxDetails, CpxEvaluations를 반환합니다.
        해당 결과가 주어진 user_id에 속하는지 확인합니다.
        """
        result = await self.db.execute(
            select(CpxResults).options(
                joinedload(CpxResults.cpx_detail),
                joinedload(CpxResults.cpx_evaluation)
            ).filter(CpxResults.result_id == result_id, CpxResults.student_id == user_id)
        )
        return result.scalars().first()

    async def get_all_cpx_results_admin(self) -> List[CpxResults]:
        """
        모든 CPX 실습 결과 목록을 반환합니다. (관리자용)
        각 결과에는 상세 정보(CpxDetails)와 평가 정보(CpxEvaluations)가 함께 로드됩니다.
        """
        result = await self.db.execute(
            select(CpxResults).options(
                joinedload(CpxResults.cpx_detail),
                joinedload(CpxResults.cpx_evaluation)
            )
        )
        return result.scalars().all()

    async def get_cpx_result_by_id(self, result_id: int) -> Optional[CpxResults]:
        """
        result_id로 특정 CPX 실습 결과를 반환합니다. (관리자용)
        각 결과에는 상세 정보(CpxDetails)와 평가 정보(CpxEvaluations)가 함께 로드됩니다.
        """
        result = await self.db.execute(
            select(CpxResults).options(
                joinedload(CpxResults.cpx_detail),
                joinedload(CpxResults.cpx_evaluation)
            ).filter(CpxResults.result_id == result_id)
        )
        return result.scalars().first()

    async def get_cpx_results_by_student_id(self, student_id: int) -> List[CpxResults]:
        """
        특정 학생 ID에 해당하는 CPX 실습 결과 목록을 반환합니다. (관리자용)
        각 결과에는 상세 정보(CpxDetails)와 평가 정보(CpxEvaluations)가 함께 로드됩니다.
        """
        result = await self.db.execute(
            select(CpxResults).options(
                joinedload(CpxResults.cpx_detail),
                joinedload(CpxResults.cpx_evaluation)
            ).filter(CpxResults.student_id == student_id)
        )
        return result.scalars().all()

    async def update_cpx_evaluation(self, result_id: int, evaluator_id: int, # evaluation_id 대신 result_id로 변경
                              overall_score: Optional[int] = None,
                              detailed_feedback: Optional[str] = None,
                              evaluation_status: Optional[str] = None) -> Optional[CpxEvaluations]:
        """
        특정 CPX 실습 결과에 대한 평가를 업데이트합니다. (관리자용)
        """
        result = await self.db.execute(
            select(CpxEvaluations).filter(CpxEvaluations.result_id == result_id) # evaluation_id 대신 result_id로 변경
        )
        evaluation = result.scalars().first()
        if not evaluation:
            return None

        evaluation.evaluator_id = evaluator_id # 평가자 변경 가능성 고려
        if overall_score is not None:
            evaluation.overall_score = overall_score
        if detailed_feedback is not None:
            evaluation.detailed_feedback = detailed_feedback
        if evaluation_status is not None:
            evaluation.evaluation_status = evaluation_status

        await self.db.commit() # await 추가
        await self.db.refresh(evaluation) # await 추가
        self.db.expunge(evaluation) # 객체 분리
        return evaluation

    async def update_cpx_result_status(self, result_id: int, new_status: str) -> Optional[CpxResults]:
        """
        특정 CPX 실습 결과의 평가 상태(evaluation_status)를 업데이트합니다. (관리자용)
        """
        result_stmt = select(CpxResults).filter(CpxResults.result_id == result_id)
        result_exec = await self.db.execute(result_stmt)
        cpx_result = result_exec.scalars().first()

        if not cpx_result:
            return None

        cpx_result.evaluation_status = new_status
        await self.db.commit()
        await self.db.refresh(cpx_result)
        return cpx_result

    async def update_cpx_details(self, result_id: int, user_id: int,
                                 memo: Optional[str] = None,
                                 system_evaluation_data: Optional[dict] = None) -> Optional[CpxDetails]:
        """
        특정 CPX 실습 결과의 상세 정보(CpxDetails)를 업데이트합니다. (학생용)
        해당 결과가 주어진 user_id에 속하는지 확인합니다.
        """
        # 먼저 CpxResults를 통해 해당 사용자의 결과인지 확인
        cpx_result_stmt = select(CpxResults).filter(CpxResults.result_id == result_id, CpxResults.student_id == user_id)
        cpx_result_exec = await self.db.execute(cpx_result_stmt)
        cpx_result = cpx_result_exec.scalars().first()

        if not cpx_result:
            return None # 해당 결과를 찾을 수 없거나 사용자 권한 없음

        # CpxDetails 조회
        details_stmt = select(CpxDetails).filter(CpxDetails.result_id == result_id)
        details_exec = await self.db.execute(details_stmt)
        cpx_details = details_exec.scalars().first()

        if not cpx_details:
            cpx_details = CpxDetails(
                result_id=result_id,
                memo=memo or "",
                system_evaluation_data=system_evaluation_data or {}
            )
            self.db.add(cpx_details)
            await self.db.flush()
        else:
            if memo is not None:
                cpx_details.memo = memo
            if system_evaluation_data is not None:
                cpx_details.system_evaluation_data = system_evaluation_data

        await self.db.commit()
        await self.db.refresh(cpx_details)
        return cpx_details


