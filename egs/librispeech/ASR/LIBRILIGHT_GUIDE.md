# LibriLight 데이터 준비 가이드

LibriLight 데이터는 이미 LibriVox 원본에 대해 전처리가 완료된 데이터셋입니다.

## 전처리가 완료된 내용들:

1. **Voice Activity Detection (VAD)**: 음성 구간 자동 탐지
2. **Signal-to-Noise Ratio (SNR)**: 신호 대 잡음비 계산
3. **메타데이터**: 화자, 책 정보, 장르 등

## JSON 메타데이터 형식:

```json
{
  "speaker": "960",           // LibriVox 화자 ID
  "snr": 5.391,              // 파일 전체 SNR 값
  "voice_activity": [         // 음성 구간 리스트 [시작, 끝] (초 단위)
    [0.4, 12.32],
    [15.1, 28.5]
  ],
  "book_meta": {             // 책 메타데이터
    "id": "319",
    "title": "History of Holland",
    "genre": ["*Non-fiction", "History"]
  }
}
```

## 사용 방법:

### 1. 데이터 압축 해제 완료 확인:
```bash
python test_librilight_format.py
```

### 2. LibriLight 매니페스트 생성:
```bash
./prepare_librilight.sh
```

### 3. 훈련에서 LibriLight 혼합 사용:
```bash
python conformer_ctc_sd_proj/train.py \
  --mix-librilight true \
  --librilight-ratio 0.3 \
  --librilight-subset medium \
  # ... 기타 훈련 옵션들
```

## 주요 특징:

- **SNR 필터링**: 낮은 품질 오디오 자동 제외
- **음성 구간 활용**: VAD로 탐지된 음성 구간만 사용
- **메타데이터 풍부**: 화자, 장르, 책 정보 활용 가능
- **효율적 처리**: 이미 전처리된 데이터로 빠른 준비

## 파라미터 조정:

- `--min-snr`: SNR 임계값 (기본: 10.0)
- `--min-duration`: 최소 세그먼트 길이 (기본: 3.0초)
- `--max-duration`: 최대 세그먼트 길이 (기본: 30.0초)
- `--librilight-ratio`: 혼합 비율 (기본: 0.3)