"""
CPX 시나리오 선택 모듈
프로그램 시작 시 시나리오를 선택하여 환자 역할을 고정
"""

from services.llm_service import LLMService


def select_cpx_scenario(llm_service: LLMService) -> bool:
    """CPX 시나리오 선택 인터페이스"""

    print("\n" + "=" * 50)
    print("🏥 CPX 가상 환자 시스템")
    print("=" * 50)

    # 사용 가능한 시나리오 표시
    scenarios = llm_service.get_available_scenarios()

    print("\n📋 사용 가능한 시나리오:")
    for scenario_id, name in scenarios.items():
        print(f"  {scenario_id}. {name}")

    print("\n시나리오를 선택해주세요 (번호 입력):")

    while True:
        try:
            choice = input("선택 > ").strip()

            if choice in scenarios:
                success = llm_service.select_scenario(choice)
                if success:
                    print(f"\n✅ 선택완료: {scenarios[choice]}")
                    print("이제 음성 대화를 시작할 수 있습니다!\n")
                    return True
                else:
                    print("❌ 시나리오 설정에 실패했습니다.")
                    return False
            else:
                print(f"❌ 잘못된 선택입니다. {list(scenarios.keys())} 중에서 선택해주세요.")

        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            return False
        except Exception as e:
            print(f"❌ 오류가 발생했습니다: {e}")


def show_scenario_info(llm_service: LLMService):
    """현재 선택된 시나리오 정보 표시"""
    if llm_service.current_scenario:
        scenario_name = llm_service.scenarios[llm_service.current_scenario]["name"]
        print(f"🎭 현재 시나리오: {scenario_name}")
    else:
        print("❗ 시나리오가 선택되지 않았습니다.")


if __name__ == "__main__":
    # 테스트용
    from services.llm_service import LLMService

    llm_service = LLMService()
    select_cpx_scenario(llm_service)
