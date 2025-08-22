from fastapi import APIRouter
router = APIRouter(prefix="/test", tags=["test"])

@router.get("/test")
async def test_for_detail():
    return {
  "student_id": 0,
  "patient_name": "뚱땡이",
  "evaluation_status": "진행중",
  "result_id": 0,
  "practice_date": "2025-08-22T04:34:41.693Z",
  "created_at": "2025-08-22T04:34:41.693Z",
  "updated_at": "2025-08-22T04:34:41.693Z",
  "cpx_detail": {
    "memo": "메모메모메모메모메모메ㅗ메모메모메ㅗ메모메모메모메모메모ㅔ모메모메모",
    "system_evaluation_data": {
  "session_id": "1_3_1755762105",
  "user_id": "1",
  "scenario_id": "3",
  "start_time": "2025-08-21T16:41:45.744595",
  "end_time": "2025-08-21T16:50:54.823835",
  "markdown": "",
  "duration_minutes": 9.151320666666667,
  "langgraph_text_analysis": {
    "evaluation_metadata": {
      "user_id": "1",
      "scenario_id": "3",
      "evaluation_date": "2025-08-21T16:50:54.825949",
      "conversation_duration_minutes": 30.5,
      "voice_recording_path": "s3로 저장",
      "conversation_transcript": "[{\"role\": \"doctor\", \"content\": \"안녕하세요 저는 박규희입니다.\", \"timestamp\": \"2025-08-21T16:42:02.630167\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"안녕하세요. 저는 나몰라입니다.\", \"timestamp\": \"2025-08-21T16:42:02.630260\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"환자분 나이와 등록번호가 어떻게 되세요?\", \"timestamp\": \"2025-08-21T16:42:16.848904\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"제 나이는 63세이고, 등록번호는 123456입니다.\", \"timestamp\": \"2025-08-21T16:42:16.848994\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"오늘 어떻게 오셨어요?\", \"timestamp\": \"2025-08-21T16:42:30.568057\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"기억력 저하로 인해 내원했어요.\", \"timestamp\": \"2025-08-21T16:42:30.568134\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"그게 언제부터 그러시는 거죠.\", \"timestamp\": \"2025-08-21T16:42:43.089965\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"6개월 전부터 기억력이 점진적으로 악화되고 있어요.\", \"timestamp\": \"2025-08-21T16:42:43.090048\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"어떻게 깜빡하시고 떨어지신지 편하게 한번 얘기해 보시겠어요?\", \"timestamp\": \"2025-08-21T16:42:59.886298\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"특별한 유발 요인 없이 기억력이 점진적으로 악화되었어요.\", \"timestamp\": \"2025-08-21T16:42:59.886383\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"어떤 식으로 까먹어요?\", \"timestamp\": \"2025-08-21T16:43:15.007427\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"물건의 위치를 잊거나 약속을 망각하는 경우가 많아요.\", \"timestamp\": \"2025-08-21T16:43:15.007506\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"비교적 최근에 일들을 까먹으시나요? 옛날 외에도 잊어버리시나요?\", \"timestamp\": \"2025-08-21T16:43:30.664045\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"주로 최근에 있었던 일들을 잊어버리지만, 장기 기억력은 상대적으로 보존되어 있어요.\", \"timestamp\": \"2025-08-21T16:43:30.664181\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"최근에 성격 변화 같은 건 없으세요?\", \"timestamp\": \"2025-08-21T16:43:45.736163\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"가벼운 우울감이 있지만 성격 변화는 크게 없어요.\", \"timestamp\": \"2025-08-21T16:43:45.736294\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"짜증이 나거나 많이 다투시거나 그런 건 없나요?\", \"timestamp\": \"2025-08-21T16:44:00.479104\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"짜증이 나거나 많이 다투는 일은 없어요. 현재는 상대적으로 차분한 편이에요.\", \"timestamp\": \"2025-08-21T16:44:00.479188\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"최근에 길을 잃어버렸다던가 다니던 뭐 이런데 잘 모르겠다거나 그런 적은 없으세요?\", \"timestamp\": \"2025-08-21T16:44:19.750526\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"길을 잃어버리거나 혼란스러워하는 증상은 없어요.요.\", \"timestamp\": \"2025-08-21T16:44:19.750598\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"직장 생활에서 기억을 못해서 문제가 되었거나 그런 거 있으세요?\", \"timestamp\": \"2025-08-21T16:44:41.547761\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"직장 생활에서 기억력으로 인해 문제가 되는 일은 없었어요.\", \"timestamp\": \"2025-08-21T16:44:41.547847\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"친구와 약속 같은 거는 깜빡한 자가 없으세요.\", \"timestamp\": \"2025-08-21T16:44:59.460095\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"가끔 약속이나 일정을 깜빡할 때가 있지만, 대부분은 기록을 통해 일정을 잊지 않도록 하고 있어요.\", \"timestamp\": \"2025-08-21T16:44:59.460175\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"가족 중에서 혹시 치매 환자가 있으신가요?\", \"timestamp\": \"2025-08-21T16:45:17.384844\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"네, 제 할아버지가 치매로 고통받으셨어요.\", \"timestamp\": \"2025-08-21T16:45:17.384944\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"몇 살 때 그러셨어요?\", \"timestamp\": \"2025-08-21T16:45:29.160870\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"할아버지께서는 60대 중반에 치매 증상이 나타나셨어요.\", \"timestamp\": \"2025-08-21T16:45:29.160958\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"환자분은 당뇨, 고혈압, 협심증 같은 혈관성 질환이 있으세요.\", \"timestamp\": \"2025-08-21T16:46:06.949426\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"고혈압이 있어요. 혈압 관리를 위해 약을 복용 중이에요. 혈관성 질환이나 당뇨는 없어요.\", \"timestamp\": \"2025-08-21T16:46:06.949513\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"몸이 많이 피곤하거나 갑상선 질환 같은 것도 없으시고요.\", \"timestamp\": \"2025-08-21T16:46:41.146996\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"몸이 많이 피곤한 것은 아니고, 갑상선 질환은 없어요.\", \"timestamp\": \"2025-08-21T16:46:41.147089\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"혹시 환각이 보이시거나 이상한 소리 한번씩 하시는 건 없으세요?\", \"timestamp\": \"2025-08-21T16:46:58.663652\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"환각이나 이상한 소리를 듣거나 보는 증상은 없어요.\", \"timestamp\": \"2025-08-21T16:46:58.663742\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"손이 떨리거나 몸이 뻣뻣하거나 느려지는 건 없으세요?\", \"timestamp\": \"2025-08-21T16:47:15.551470\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"손이 떨리거나 몸이 뻣뻣하거나 느려지는 증상은 없어요.\", \"timestamp\": \"2025-08-21T16:47:15.551552\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"평소에 술 많이 드시는 편이세요.\", \"timestamp\": \"2025-08-21T16:47:30.604382\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"주 1-2회 정도 사회적으로 음주를 하지만, 많이 마시는 편은 아니에요.\", \"timestamp\": \"2025-08-21T16:47:30.604490\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"혹시 머리를 다치시거나 뇌염 같은 뇌질환 아론적 없으세요?\", \"timestamp\": \"2025-08-21T16:47:52.007389\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"머리를 다치거나 뇌염 같은 뇌질환은 없어요.\", \"timestamp\": \"2025-08-21T16:47:52.007478\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"알겠습니다 그럼 이제 신체 진찰을 하도록 하겠습니다.\", \"timestamp\": \"2025-08-21T16:48:09.853356\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"네, 알겠습니다. 신체 진찰을 받겠습니다.\", \"timestamp\": \"2025-08-21T16:48:09.853436\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"검사는 mmse 검사를 시행하겠습니다.\", \"timestamp\": \"2025-08-21T16:48:25.082385\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"네, 알겠습니다. MMSE 검사를 받겠습니다.\", \"timestamp\": \"2025-08-21T16:48:25.082471\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"이제 진찰을 끝났구요 혹시 걱정되는 거 있으세요?\", \"timestamp\": \"2025-08-21T16:48:41.259747\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"아무 걱정거리가 없어요. 현재는 걱정되는 것이 없어요.\", \"timestamp\": \"2025-08-21T16:48:41.259834\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"일단은 가족력이 있으시고 또 기억력이 떨어지는 것 때문에.\", \"timestamp\": \"2025-08-21T16:48:57.646455\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"가족력이 있어서 약간은 걱정되긴 하지만, 현재는 크게 걱정되는 것이 없어요.\", \"timestamp\": \"2025-08-21T16:48:57.646536\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"치매 가장 흔한 유형인 아이츠라이머.\", \"timestamp\": \"2025-08-21T16:49:07.680273\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"네, 알츠하이머 치매가 가장 흔한 유형이 맞아요.\", \"timestamp\": \"2025-08-21T16:49:07.680421\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"가능성을 고려해야 될 것 같습니다.\", \"timestamp\": \"2025-08-21T16:49:20.925947\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"네, 알츠하이머 치매의 가능성을 고려해야 할 것 같아요.\", \"timestamp\": \"2025-08-21T16:49:20.926030\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"하지만 우울증 때문에 가성치의.\", \"timestamp\": \"2025-08-21T16:49:39.134615\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"우울증도 고려해봐야 하지만, 알츠하이머 치매 가능성도 염두에 두어야 할 것 같아요.\", \"timestamp\": \"2025-08-21T16:49:39.134703\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"또 조금 가능성이 높지는 않지만 고혈압이 있어서 혈관성 치매 가능성도 고려해야 할 것 같습니다.\", \"timestamp\": \"2025-08-21T16:50:04.018528\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"고혈압으로 인한 혈관성 치매 가능성도 함께 고려해야 할 것 같아요. 여러 가능성을 함께 고려하는 것이 중요할 것 같아요.\", \"timestamp\": \"2025-08-21T16:50:04.018660\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"혹시 또 다른 궁금한 거 있으세요?\", \"timestamp\": \"2025-08-21T16:50:21.100627\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"잘 모르겠어요.\", \"timestamp\": \"2025-08-21T16:50:21.100743\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"없으시면 검사 후에 뵙도록 하겠습니다. 조심해서 가세요.\", \"timestamp\": \"2025-08-21T16:50:37.796974\", \"emotion\": null}, {\"role\": \"patient\", \"content\": \"네, 감사합니다. 검사 후에 뵙겠습니다. 좋은 하루 되세요.\", \"timestamp\": \"2025-08-21T16:50:37.797057\", \"emotion\": null}, {\"role\": \"doctor\", \"content\": \"안녕히 가세요.\", \"timestamp\": \"2025-08-21T16:50:54.823802\", \"emotion\": {\"predicted_emotion\": \"Kind\", \"confidence\": 0.5, \"emotion_scores\": {\"Kind\": 0.5, \"Anxious\": 0.3, \"Dry\": 0.2}}}]"
    },
    "scores": {
      "total_score": 74.0,
      "completion_rate": 0.68,
      "quality_score": 8,
      "grade": "C+"
    },
    "feedback": {
      "strengths": [
        "병력 청취 영역 우수 (71.4%)",
        "신체 진찰 영역 우수 (100.0%)",
        "의사는 환자의 기억력 저하에 대한 다양한 가능성을 고려하고, 치매와 관련된 가족력, 혈관성 질환 등을 체계적으로 질문함.",
        "의학 용어 사용이 적절하며, 진단적 접근이 논리적임.",
        "환자에게 공감하며 친절한 태도로 대화를 이끌어감."
      ],
      "improvements": [
        "일부 질문이 반복적이거나 명확하지 않아 환자가 혼란스러울 수 있음.",
        "환자가 이해하기 쉬운 언어로 설명하는 데 더 신경 쓸 필요가 있음.",
        "대화 흐름이 다소 끊기는 부분이 있어 자연스러운 흐름을 유지할 필요가 있음."
      ],
      "overall_analysis": "RAG 기반 평가 결과 74.0% 완성"
    },
    "conversation_summary": {
      "total_questions": 31,
      "duration_minutes": 9.151320666666667
    },
    "detailed_analysis": {
      "completeness": {
        "category": "기억력 저하",
        "overall_completeness": 0.68,
        "areas_evaluation": {
          "history_taking": {
            "area_name": "병력 청취",
            "total_guidelines": 14,
            "completed_guidelines": 10,
            "completion_rate": 0.7142857142857143,
            "guideline_evaluations": [
              {
                "guideline_item": "O (Onset) - 발병 시기",
                "completed": True,
                "evidence": [
                  "의사: 그게 언제부터 그러시는 거죠.",
                  "환자: 6개월 전부터 기억력이 점진적으로 악화되고 있어요."
                ],
                "required_action": []
              },
              {
                "guideline_item": "D (Duration) - 지속시간/변동성",
                "completed": True,
                "evidence": [
                  "환자: 6개월 전부터 기억력이 점진적으로 악화되고 있어요."
                ],
                "required_action": []
              },
              {
                "guideline_item": "Co (Course) - 경과",
                "completed": True,
                "evidence": [
                  "환자: 6개월 전부터 기억력이 점진적으로 악화되고 있어요."
                ],
                "required_action": []
              },
              {
                "guideline_item": "Ex (Experience) - 과거 경험",
                "completed": False,
                "evidence": [],
                "required_action": [
                  "이전에도 이런 적이 있나요?",
                  "당시 치료를 받았나요?"
                ]
              },
              {
                "guideline_item": "C (Character) - 증상 특징",
                "completed": True,
                "evidence": [
                  "의사: 어떤 식으로 까먹어요?",
                  "환자: 물건의 위치를 잊거나 약속을 망각하는 경우가 많아요.",
                  "의사: 비교적 최근에 일들을 까먹으시나요? 옛날 외에도 잊어버리시나요?",
                  "환자: 주로 최근에 있었던 일들을 잊어버리지만, 장기 기억력은 상대적으로 보존되어 있어요.",
                  "의사: 최근에 길을 잃어버렸다던가 다니던 뭐 이런데 잘 모르겠다거나 그런 적은 없으세요?",
                  "환자: 길을 잃어버리거나 혼란스러워하는 증상은 없어요."
                ],
                "required_action": []
              },
              {
                "guideline_item": "A (Associated symptom) - 동반 증상 (감별진단별)",
                "completed": True,
                "evidence": [
                  "의사: 최근에 성격 변화 같은 건 없으세요?",
                  "환자: 가벼운 우울감이 있지만 성격 변화는 크게 없어요.",
                  "의사: 짜증이 나거나 많이 다투시거나 그런 건 없나요?",
                  "환자: 짜증이 나거나 많이 다투는 일은 없어요. 현재는 상대적으로 차분한 편이에요.",
                  "의사: 손이 떨리거나 몸이 뻣뻣하거나 느려지는 건 없으세요?",
                  "환자: 손이 떨리거나 몸이 뻣뻣하거나 느려지는 증상은 없어요."
                ],
                "required_action": []
              },
              {
                "guideline_item": "F (Factor) - 악화/완화 요인",
                "completed": False,
                "evidence": [],
                "required_action": [
                  "스트레스를 많이 받으시나요?"
                ]
              },
              {
                "guideline_item": "E (Exam) - 이전 검사/건강검진",
                "completed": False,
                "evidence": [],
                "required_action": [
                  "이전 건강 검진에서 이상 소견은 없었나요?"
                ]
              },
              {
                "guideline_item": "외상력",
                "completed": True,
                "evidence": [
                  "의사: 혹시 머리를 다치시거나 뇌염 같은 뇌질환 아론적 없으세요?",
                  "환자: 머리를 다치거나 뇌염 같은 뇌질환은 없어요."
                ],
                "required_action": []
              },
              {
                "guideline_item": "과거력",
                "completed": True,
                "evidence": [
                  "의사: 환자분은 당뇨, 고혈압, 협심증 같은 혈관성 질환이 있으세요.",
                  "환자: 고혈압이 있어요. 혈압 관리를 위해 약을 복용 중이에요. 혈관성 질환이나 당뇨는 없어요."
                ],
                "required_action": []
              },
              {
                "guideline_item": "약물력",
                "completed": True,
                "evidence": [
                  "환자: 고혈압이 있어요. 혈압 관리를 위해 약을 복용 중이에요."
                ],
                "required_action": []
              },
              {
                "guideline_item": "사회력",
                "completed": True,
                "evidence": [
                  "의사: 평소에 술 많이 드시는 편이세요.",
                  "환자: 주 1-2회 정도 사회적으로 음주를 하지만, 많이 마시는 편은 아니에요."
                ],
                "required_action": []
              },
              {
                "guideline_item": "가족력",
                "completed": True,
                "evidence": [
                  "의사: 가족 중에서 혹시 치매 환자가 있으신가요?",
                  "환자: 네, 제 할아버지가 치매로 고통받으셨어요.",
                  "의사: 몇 살 때 그러셨어요?",
                  "환자: 할아버지께서는 60대 중반에 치매 증상이 나타나셨어요."
                ],
                "required_action": []
              },
              {
                "guideline_item": "여성력 (해당시)",
                "completed": False,
                "evidence": [],
                "required_action": [
                  "LMP / 규칙적 / 주기 / 폐경"
                ]
              }
            ]
          },
          "physical_examination": {
            "area_name": "신체 진찰",
            "total_guidelines": 2,
            "completed_guidelines": 2,
            "completion_rate": 1.0,
            "guideline_evaluations": [
              {
                "guideline_item": "진찰 준비",
                "completed": True,
                "evidence": [
                  "의사: 알겠습니다 그럼 이제 신체 진찰을 하도록 하겠습니다.",
                  "환자: 네, 알겠습니다. 신체 진찰을 받겠습니다."
                ],
                "required_action": []
              },
              {
                "guideline_item": "검사",
                "completed": True,
                "evidence": [
                  "의사: 검사는 mmse 검사를 시행하겠습니다.",
                  "환자: 네, 알겠습니다. MMSE 검사를 받겠습니다."
                ],
                "required_action": []
              }
            ]
          },
          "patient_education": {
            "area_name": "환자 교육",
            "total_guidelines": 6,
            "completed_guidelines": 3,
            "completion_rate": 0.5,
            "guideline_evaluations": [
              {
                "guideline_item": "공감",
                "completed": False,
                "evidence": [],
                "required_action": [
                  "기억력이 자꾸 나빠져서 많이 걱정되셨을 것 같습니다."
                ]
              },
              {
                "guideline_item": "추정 진단",
                "completed": True,
                "evidence": [
                  "의사: 일단은 가족력이 있으시고 또 기억력이 떨어지는 것 때문에.",
                  "환자: 가족력이 있어서 약간은 걱정되긴 하지만, 현재는 크게 걱정되는 것이 없어요.",
                  "의사: 치매 가장 흔한 유형인 아이츠라이머.",
                  "환자: 네, 알츠하이머 치매가 가장 흔한 유형이 맞아요.",
                  "의사: 가능성을 고려해야 될 것 같습니다.",
                  "환자: 네, 알츠하이머 치매의 가능성을 고려해야 할 것 같아요."
                ],
                "required_action": []
              },
              {
                "guideline_item": "감별 진단",
                "completed": True,
                "evidence": [
                  "의사: 하지만 우울증 때문에 가성치의.",
                  "환자: 우울증도 고려해봐야 하지만, 알츠하이머 치매 가능성도 염두에 두어야 할 것 같아요.",
                  "의사: 또 조금 가능성이 높지는 않지만 고혈압이 있어서 혈관성 치매 가능성도 고려해야 할 것 같습니다.",
                  "환자: 고혈압으로 인한 혈관성 치매 가능성도 함께 고려해야 할 것 같아요. 여러 가능성을 함께 고려하는 것이 중요할 것 같아요."
                ],
                "required_action": []
              },
              {
                "guideline_item": "검사 계획",
                "completed": False,
                "evidence": [],
                "required_action": [
                  "정확한 원인 감별을 위해 뇌 MRI와 뇌 PET 검사를 촬영하도록 하고, 갑상샘 기능을 보기 위한 혈액검사가 필요합니다."
                ]
              },
              {
                "guideline_item": "치료 및 교육",
                "completed": False,
                "evidence": [],
                "required_action": [
                  "알츠하이머 치매로 진단되는 경우 인지 기능 개선 약물로 치료를 시작할 것입니다.",
                  "치매의 치료에 가족 분들의 도움이 많이 필요합니다.",
                  "국가에서 운영하는 치매 지원 센터가 있는데, 구 단위로 있으며, 이곳에서 검사도 가능하고 치료비나 기타 활동 지원사업도 해주고 있으니 필요하시면 소개해드리겠습니다."
                ]
              },
              {
                "guideline_item": "마무리 질문",
                "completed": True,
                "evidence": [
                  "의사: 혹시 또 다른 궁금한 거 있으세요?",
                  "환자: 잘 모르겠어요.",
                  "의사: 없으시면 검사 후에 뵙도록 하겠습니다. 조심해서 가세요.",
                  "환자: 네, 감사합니다. 검사 후에 뵙겠습니다. 좋은 하루 되세요."
                ],
                "required_action": []
              }
            ]
          }
        },
        "total_completed_items": 3,
        "total_missing_items": 2,
        "evaluation_method": "rag_three_areas"
      },
      "quality": {
        "medical_accuracy": 8,
        "communication_efficiency": 7,
        "professionalism": 8,
        "scenario_appropriateness": 8,
        "overall_quality_score": 8,
        "quality_strengths": [
          "의사는 환자의 기억력 저하에 대한 다양한 가능성을 고려하고, 치매와 관련된 가족력, 혈관성 질환 등을 체계적으로 질문함.",
          "의학 용어 사용이 적절하며, 진단적 접근이 논리적임.",
          "환자에게 공감하며 친절한 태도로 대화를 이끌어감."
        ],
        "quality_improvements": [
          "일부 질문이 반복적이거나 명확하지 않아 환자가 혼란스러울 수 있음.",
          "환자가 이해하기 쉬운 언어로 설명하는 데 더 신경 쓸 필요가 있음.",
          "대화 흐름이 다소 끊기는 부분이 있어 자연스러운 흐름을 유지할 필요가 있음."
        ]
      },
      "comprehensive": {
        "final_score": 7.4,
        "grade": "C+",
        "detailed_feedback": {
          "strengths": [
            "병력 청취 영역 우수 (71.4%)",
            "신체 진찰 영역 우수 (100.0%)",
            "의사는 환자의 기억력 저하에 대한 다양한 가능성을 고려하고, 치매와 관련된 가족력, 혈관성 질환 등을 체계적으로 질문함.",
            "의학 용어 사용이 적절하며, 진단적 접근이 논리적임.",
            "환자에게 공감하며 친절한 태도로 대화를 이끌어감."
          ],
          "improvements": [
            "일부 질문이 반복적이거나 명확하지 않아 환자가 혼란스러울 수 있음.",
            "환자가 이해하기 쉬운 언어로 설명하는 데 더 신경 쓸 필요가 있음.",
            "대화 흐름이 다소 끊기는 부분이 있어 자연스러운 흐름을 유지할 필요가 있음."
          ],
          "overall_analysis": "RAG 기반 평가 결과 74.0% 완성"
        }
      }
    },
    "evaluation_method": "3단계 의학적 분석",
    "system_info": {
      "version": "v2.0",
      "evaluation_steps": 3
    }
  },
  "conversation_entries": [
    {
      "timestamp": "2025-08-21T16:42:02.630167",
      "text": "안녕하세요 저는 박규희입니다.",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:42:02.630260",
      "text": "안녕하세요. 저는 나몰라입니다.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:42:16.848904",
      "text": "환자분 나이와 등록번호가 어떻게 되세요?",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:42:16.848994",
      "text": "제 나이는 63세이고, 등록번호는 123456입니다.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:42:30.568057",
      "text": "오늘 어떻게 오셨어요?",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:42:30.568134",
      "text": "기억력 저하로 인해 내원했어요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:42:43.089965",
      "text": "그게 언제부터 그러시는 거죠.",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:42:43.090048",
      "text": "6개월 전부터 기억력이 점진적으로 악화되고 있어요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:42:59.886298",
      "text": "어떻게 깜빡하시고 떨어지신지 편하게 한번 얘기해 보시겠어요?",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:42:59.886383",
      "text": "특별한 유발 요인 없이 기억력이 점진적으로 악화되었어요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:43:15.007427",
      "text": "어떤 식으로 까먹어요?",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:43:15.007506",
      "text": "물건의 위치를 잊거나 약속을 망각하는 경우가 많아요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:43:30.664045",
      "text": "비교적 최근에 일들을 까먹으시나요? 옛날 외에도 잊어버리시나요?",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:43:30.664181",
      "text": "주로 최근에 있었던 일들을 잊어버리지만, 장기 기억력은 상대적으로 보존되어 있어요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:43:45.736163",
      "text": "최근에 성격 변화 같은 건 없으세요?",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:43:45.736294",
      "text": "가벼운 우울감이 있지만 성격 변화는 크게 없어요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:44:00.479104",
      "text": "짜증이 나거나 많이 다투시거나 그런 건 없나요?",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:44:00.479188",
      "text": "짜증이 나거나 많이 다투는 일은 없어요. 현재는 상대적으로 차분한 편이에요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:44:19.750526",
      "text": "최근에 길을 잃어버렸다던가 다니던 뭐 이런데 잘 모르겠다거나 그런 적은 없으세요?",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:44:19.750598",
      "text": "길을 잃어버리거나 혼란스러워하는 증상은 없어요.요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:44:41.547761",
      "text": "직장 생활에서 기억을 못해서 문제가 되었거나 그런 거 있으세요?",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:44:41.547847",
      "text": "직장 생활에서 기억력으로 인해 문제가 되는 일은 없었어요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:44:59.460095",
      "text": "친구와 약속 같은 거는 깜빡한 자가 없으세요.",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:44:59.460175",
      "text": "가끔 약속이나 일정을 깜빡할 때가 있지만, 대부분은 기록을 통해 일정을 잊지 않도록 하고 있어요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:45:17.384844",
      "text": "가족 중에서 혹시 치매 환자가 있으신가요?",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:45:17.384944",
      "text": "네, 제 할아버지가 치매로 고통받으셨어요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:45:29.160870",
      "text": "몇 살 때 그러셨어요?",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:45:29.160958",
      "text": "할아버지께서는 60대 중반에 치매 증상이 나타나셨어요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:46:06.949426",
      "text": "환자분은 당뇨, 고혈압, 협심증 같은 혈관성 질환이 있으세요.",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:46:06.949513",
      "text": "고혈압이 있어요. 혈압 관리를 위해 약을 복용 중이에요. 혈관성 질환이나 당뇨는 없어요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:46:41.146996",
      "text": "몸이 많이 피곤하거나 갑상선 질환 같은 것도 없으시고요.",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:46:41.147089",
      "text": "몸이 많이 피곤한 것은 아니고, 갑상선 질환은 없어요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:46:58.663652",
      "text": "혹시 환각이 보이시거나 이상한 소리 한번씩 하시는 건 없으세요?",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:46:58.663742",
      "text": "환각이나 이상한 소리를 듣거나 보는 증상은 없어요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:47:15.551470",
      "text": "손이 떨리거나 몸이 뻣뻣하거나 느려지는 건 없으세요?",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:47:15.551552",
      "text": "손이 떨리거나 몸이 뻣뻣하거나 느려지는 증상은 없어요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:47:30.604382",
      "text": "평소에 술 많이 드시는 편이세요.",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:47:30.604490",
      "text": "주 1-2회 정도 사회적으로 음주를 하지만, 많이 마시는 편은 아니에요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:47:52.007389",
      "text": "혹시 머리를 다치시거나 뇌염 같은 뇌질환 아론적 없으세요?",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:47:52.007478",
      "text": "머리를 다치거나 뇌염 같은 뇌질환은 없어요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:48:09.853356",
      "text": "알겠습니다 그럼 이제 신체 진찰을 하도록 하겠습니다.",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:48:09.853436",
      "text": "네, 알겠습니다. 신체 진찰을 받겠습니다.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:48:25.082385",
      "text": "검사는 mmse 검사를 시행하겠습니다.",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:48:25.082471",
      "text": "네, 알겠습니다. MMSE 검사를 받겠습니다.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:48:41.259747",
      "text": "이제 진찰을 끝났구요 혹시 걱정되는 거 있으세요?",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:48:41.259834",
      "text": "아무 걱정거리가 없어요. 현재는 걱정되는 것이 없어요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:48:57.646455",
      "text": "일단은 가족력이 있으시고 또 기억력이 떨어지는 것 때문에.",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:48:57.646536",
      "text": "가족력이 있어서 약간은 걱정되긴 하지만, 현재는 크게 걱정되는 것이 없어요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:49:07.680273",
      "text": "치매 가장 흔한 유형인 아이츠라이머.",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:49:07.680421",
      "text": "네, 알츠하이머 치매가 가장 흔한 유형이 맞아요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:49:20.925947",
      "text": "가능성을 고려해야 될 것 같습니다.",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:49:20.926030",
      "text": "네, 알츠하이머 치매의 가능성을 고려해야 할 것 같아요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:49:39.134615",
      "text": "하지만 우울증 때문에 가성치의.",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:49:39.134703",
      "text": "우울증도 고려해봐야 하지만, 알츠하이머 치매 가능성도 염두에 두어야 할 것 같아요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:50:04.018528",
      "text": "또 조금 가능성이 높지는 않지만 고혈압이 있어서 혈관성 치매 가능성도 고려해야 할 것 같습니다.",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:50:04.018660",
      "text": "고혈압으로 인한 혈관성 치매 가능성도 함께 고려해야 할 것 같아요. 여러 가능성을 함께 고려하는 것이 중요할 것 같아요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:50:21.100627",
      "text": "혹시 또 다른 궁금한 거 있으세요?",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:50:21.100743",
      "text": "잘 모르겠어요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:50:37.796974",
      "text": "없으시면 검사 후에 뵙도록 하겠습니다. 조심해서 가세요.",
      "role": "doctor",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:50:37.797057",
      "text": "네, 감사합니다. 검사 후에 뵙겠습니다. 좋은 하루 되세요.",
      "role": "patient",
      "emotion": None
    },
    {
      "timestamp": "2025-08-21T16:50:54.823802",
      "text": "안녕히 가세요.",
      "role": "doctor",
      "emotion": {
        "predicted_emotion": "Kind",
        "confidence": 0.5,
        "emotion_scores": {
          "Kind": 0.5,
          "Anxious": 0.3,
          "Dry": 0.2
        }
      }
    }
  ]
},
    "detail_id": 0,
    "result_id": 0,
    "last_updated_at": "2025-08-22T04:34:41.693Z"
  },
  "cpx_evaluation": {
    "overall_score": 0,
    "detailed_feedback": "피드백드개브갭드개븓개브대그배드개브댁배드개븓개ㅡ배급대ㅡㄱ배드개브대그배드ㅐㅂ드개브",
    "evaluation_status": "평가대기",
    "evaluation_id": 0,
    "result_id": 0,
    "evaluator_id": 0,
    "evaluation_date": "2025-08-22T04:34:41.693Z",
    "created_at": "2025-08-22T04:34:41.693Z",
    "updated_at": "2025-08-22T04:34:41.693Z"
  }
}