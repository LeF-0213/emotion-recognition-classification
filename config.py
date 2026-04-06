import os
from pathlib import Path

SEED = 2026

ROOT = Path(__file__).resolve().parent

# 정리 스크립트 실행 후 기본 데이터 위치
DATA_ROOT = (ROOT / "dataset").resolve()
SPEC_DIR = DATA_ROOT / "spectrograms"
TEXT_CSV = DATA_ROOT / "texts.csv"
LABEL_CSV = DATA_ROOT / "labels.csv"

OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

EMOTION_CLASSES = ["happy", "sad", "angry", "neutral"]
LABEL2IDX = {e: i for i, e in enumerate(EMOTION_CLASSES)}
IDX2LABEL = {i: e for e, i in LABEL2IDX.items()}
NUM_CLASSES = len(EMOTION_CLASSES)

COL_ID = "id"
COL_TEXT = "text"
COL_EMOTION = "emotion"

TEXT_MODEL_NAME = "bert-base-uncased"
MAX_TEXT_LEN = 128
IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 32
NUM_EPOCHS = 15
LR = 2e-5               # 학습률(학습 속도)
WEIGHT_DECAY = 0.01     # 가중치 감소(뉴런 값이 너무 커지는 것을 방지)
DROPOUT = 0.3           # 과적합 방지(무작위로 뉴런 비활성화)

TEXT_FREEZE_LAYERS = 6     # 텍스트 모델 레이어 동결 (하위 레이어는 기초 정보를 추출하기 때문에 동결)
IMAGE_FREEZE_RATIO = 0.7  # 이미지 모델 레이어 동결 

EARLY_STOP_PATIENCE = 5  # 조기 종료 기간
GRAD_CLIP = 1.0          # 그레디언트 클리핑(그레디언트 값이 너무 커지는 것을 방지)
NUM_WORKERS = 2          # 데이터 로더 작업 수(CPU가 데이터를 얼마나 병렬로 준비할지)

IMAGE_BACKBONE = "efficientnet_b0"

MIXUP_ALPHA = 0.4       # 두 레이블을 섞어서 학습(레이블 값을 부드럽게 만들어 학습 안정성을 높임)
LABEL_SMOOTHING = 0.1   # 레이블 스무딩(레이블 값을 부드럽게 만들어 학습 안정성을 높임)

# 아랍어 텍스트 → 영어 (BERT와 맞추고 싶을 때). 음성은 여전히 아랍어 발화 기반 스펙트로그램.
TRANSLATE_AR_TO_EN = False
TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-ar-en"
TRANSLATION_BATCH_SIZE = 8
TRANSLATION_MAX_CHARS = 400