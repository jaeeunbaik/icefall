#!/usr/bin/env python3
"""
Self-Distillation을 위한 Augmentation 비율 권장 설정
MUSAN과 SpecAugment의 적절한 조합을 분석합니다.
"""

def analyze_augmentation_strategies():
    """Self-distillation을 위한 augmentation 전략 분석"""
    
    print("🎯 Self-Distillation Augmentation 설정 가이드")
    print("="*60)
    
    print("\n📊 현재 설정 현황:")
    print("✅ MUSAN: CutMix(p=0.5, snr=(10, 20))")
    print("✅ SpecAugment: time_warp=80, frame_masks=10/2, feature_masks=2")
    
    print("\n🔬 Self-Distillation의 핵심 원리:")
    print("1. Clean Input (Teacher Signal) ← 깨끗한 데이터")
    print("2. Augmented Input (Student Signal) ← 강하게 augmented 데이터")
    print("3. Knowledge Transfer: Clean → Augmented")
    
    print("\n" + "="*60)
    print("🎛️  권장 Augmentation 설정:")
    print("="*60)
    
    print("\n💡 Strategy 1: Conservative (안정적 학습)")
    print("📌 MUSAN:")
    print("   - Probability: 0.3-0.4 (30-40%)")
    print("   - SNR: (15, 25) dB")
    print("   - 이유: 너무 강한 노이즈는 teacher-student gap 증가")
    
    print("\n📌 SpecAugment:")
    print("   - Time Warp: 40-60")
    print("   - Frame Masks: 8-10개")
    print("   - Feature Masks: 2개")
    print("   - 이유: 적당한 masking으로 overfitting 방지")
    
    print("\n💡 Strategy 2: Aggressive (강한 regularization)")
    print("📌 MUSAN:")
    print("   - Probability: 0.6-0.7 (60-70%)")
    print("   - SNR: (10, 20) dB")
    print("   - 이유: 강한 augmentation으로 robustness 향상")
    
    print("\n📌 SpecAugment:")
    print("   - Time Warp: 80-100")
    print("   - Frame Masks: 12-15개")
    print("   - Feature Masks: 3개")
    print("   - 이유: 강한 masking으로 지식 전이 효과 극대화")
    
    print("\n💡 Strategy 3: Balanced (균형적 접근)")
    print("📌 MUSAN:")
    print("   - Probability: 0.5 (50%)")
    print("   - SNR: (12, 22) dB")
    print("   - 이유: 현재 설정과 유사, 검증된 설정")
    
    print("\n📌 SpecAugment:")
    print("   - Time Warp: 80")
    print("   - Frame Masks: 10개")
    print("   - Feature Masks: 2개")
    print("   - 이유: 표준적인 설정, 많은 논문에서 사용")

def recommend_starting_configuration():
    """시작하기 좋은 설정 추천"""
    
    print("\n" + "="*60)
    print("🚀 추천 시작 설정 (Strategy 3 기반):")
    print("="*60)
    
    print("\n📝 train_sd.sh 수정 사항:")
    print("""
# MUSAN 설정 (현재와 동일)
--enable-musan=True
# CutMix probability는 코드에서 0.5로 고정됨

# SpecAugment 설정 (현재와 동일)
--enable-spec-aug=True
--spec-aug-time-warp-factor=80
""")
    
    print("\n💡 실험적 조정 가능한 부분:")
    print("1. MUSAN SNR 범위 조정")
    print("2. CutMix probability 변경 (코드 수정 필요)")
    print("3. SpecAugment 강도 조정")
    
    print("\n🔧 세밀한 조정을 위한 코드 수정:")
    print("asr_datamodule.py의 CutMix 설정 부분:")
    print("   CutMix(cuts=cuts_musan, p=0.5, snr=(10, 20), preserve_id=True)")
    print("   → p값과 snr 범위를 실험적으로 조정 가능")

def suggest_experimental_progression():
    """실험 진행 순서 제안"""
    
    print("\n" + "="*60)
    print("🧪 실험 진행 순서:")
    print("="*60)
    
    print("\n📅 Phase 1: Baseline (현재 설정)")
    print("   - MUSAN: p=0.5, snr=(10,20)")
    print("   - SpecAugment: time_warp=80, 표준 설정")
    print("   - 목표: 기본 성능 확인")
    
    print("\n📅 Phase 2: Conservative Tuning")
    print("   - MUSAN: p=0.3, snr=(15,25)")
    print("   - SpecAugment: time_warp=60")
    print("   - 목표: 안정성 vs 성능 trade-off 확인")
    
    print("\n📅 Phase 3: Aggressive Tuning")
    print("   - MUSAN: p=0.7, snr=(8,18)")
    print("   - SpecAugment: time_warp=100")
    print("   - 목표: 최대 regularization 효과 확인")
    
    print("\n📊 각 실험별 모니터링 지표:")
    print("   ✅ Clean vs Augmented Loss Gap")
    print("   ✅ Validation WER")
    print("   ✅ Training Stability")
    print("   ✅ Convergence Speed")

def current_setting_analysis():
    """현재 설정 분석"""
    
    print("\n" + "="*60)
    print("🔍 현재 설정 분석:")
    print("="*60)
    
    print("\n현재 asr_datamodule.py 설정:")
    print("📌 MUSAN CutMix:")
    print("   - Probability: 0.5 (50% of batches)")
    print("   - SNR Range: 10-20 dB")
    print("   - 평가: 적당히 강한 노이즈, 표준적 설정")
    
    print("\n📌 SpecAugment:")
    print("   - Time Warp: 80")
    print("   - Frame Masks: 10개 (또는 2개, Lhotse 버전별)")
    print("   - Feature Masks: 2개")
    print("   - 평가: 표준적이고 안정적인 설정")
    
    print("\n✅ 장점:")
    print("   - 검증된 설정값들")
    print("   - 안정적인 학습 예상")
    print("   - Self-distillation과 호환성 좋음")
    
    print("\n🤔 개선 가능성:")
    print("   - MUSAN probability를 0.6-0.7로 증가 가능")
    print("   - SNR 하한을 8-9 dB로 낮춰 더 강한 노이즈 적용 가능")
    print("   - SpecAugment time_warp를 100으로 증가 가능")
    
    print("\n🎯 결론:")
    print("현재 설정으로 시작하되, 학습 과정에서 다음 관찰:")
    print("   - Clean vs Augmented input의 loss 차이")
    print("   - Teacher model의 수렴 속도")
    print("   - Validation 성능 향상 정도")

if __name__ == "__main__":
    analyze_augmentation_strategies()
    recommend_starting_configuration()
    suggest_experimental_progression()
    current_setting_analysis()
    
    print("\n" + "="*60)
    print("📋 실행 권장사항:")
    print("="*60)
    print("1. 현재 설정으로 첫 실험 시작 ✅")
    print("2. 10-20 epoch 후 loss gap 모니터링")
    print("3. 필요시 MUSAN probability 조정")
    print("4. SpecAugment 강도는 마지막에 조정")
    print("5. 각 설정별 3-5 에포크 검증 후 다음 단계")
    print("\n🚀 지금 바로 train_sd.sh로 시작 가능합니다!")
