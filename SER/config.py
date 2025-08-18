class Config:
  def __init__(self):
    EMOTION_LABELS = ["Anxious", "Dry", "Kind"]
    LABEL2ID = {label: i for i, label in enumerate(EMOTION_LABELS)}
    ID2LABEL = {i: label for i, label in enumerate(EMOTION_LABELS)}

    MODEL_NAME = "kresnik/wav2vec2-large-xlsr-korean"
    SAMPLE_RATE = 16000
    MAX_DURATION = 10.0