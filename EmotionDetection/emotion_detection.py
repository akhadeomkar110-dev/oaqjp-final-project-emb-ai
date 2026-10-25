"""Client for the Watson NLP emotion prediction service."""

import requests

EMOTION_SERVICE_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
EMOTION_MODEL_ID = "emotion_aggregated-workflow_lang_en_stock"
EMOTION_NAMES = ("anger", "disgust", "fear", "joy", "sadness")

def emotion_detector(text_to_analyze):
    """Return five emotion scores and the dominant emotion."""
    headers = {"grpc-metadata-mm-model-id": EMOTION_MODEL_ID}
    payload = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(
        EMOTION_SERVICE_URL, json=payload, headers=headers, timeout=30
    )
    if response.status_code == 400:
        return None
    response.raise_for_status()
    emotions = response.json()["emotionPredictions"][0]["emotion"]
    result = {name: emotions[name] for name in EMOTION_NAMES}
    result["dominant_emotion"] = max(EMOTION_NAMES, key=result.__getitem__)
    return result
