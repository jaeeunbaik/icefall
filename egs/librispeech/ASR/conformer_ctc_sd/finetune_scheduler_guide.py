#!/usr/bin/env python3
"""
Self-Distillation Fine-tuning을 위한 Learning Rate Scheduler 가이드
수렴된 모델에서 self-distillation을 적용할 때의 최적 스케줄러 설정
"""

def analyze_scheduler_options():
    """Fine-tuning에 적합한 스케줄러 옵션들을 분석"""
    
    print("🎯 Self-Distillation Fine-tuning용 스케줄러 가이드")
    print("="*65)
    
    print("\n🔍 현재 상황 분석:")
    print("✅ 수렴된 모델에서 시작 (pre-trained)")
    print("✅ Self-distillation 적용 (fine-tuning)")
    print("✅ 안정적인 학습 필요")
    print("❌ Noam scheduler: 처음부터 학습용, fine-tuning에 부적합")
    
    print("\n" + "="*65)
    print("📊 추천 스케줄러 옵션들:")
    print("="*65)
    
    print("\n💡 Option 1: Constant LR (가장 안전)")
    print("📌 특징:")
    print("   - 고정된 학습률 사용")
    print("   - 가장 안정적이고 예측 가능")
    print("   - Fine-tuning의 표준 방법")
    print("📌 추천 설정:")
    print("   - Learning Rate: 1e-5 ~ 5e-5")
    print("   - Weight Decay: 1e-6 ~ 1e-4")
    print("   - No warmup needed")
    
    print("\n💡 Option 2: Cosine Annealing (부드러운 감소)")
    print("📌 특징:")
    print("   - 코사인 함수로 부드럽게 감소")
    print("   - 마지막에 매우 작은 lr로 수렴")
    print("   - 일반화 성능 향상에 도움")
    print("📌 추천 설정:")
    print("   - Initial LR: 2e-5")
    print("   - T_max: total_epochs (전체 에포크 수)")
    print("   - eta_min: 1e-6")
    
    print("\n💡 Option 3: Step LR (단계별 감소)")
    print("📌 특징:")
    print("   - 특정 에포크마다 lr을 고정 비율로 감소")
    print("   - 제어하기 쉽고 해석 용이")
    print("   - Validation loss plateau에 따라 조정 가능")
    print("📌 추천 설정:")
    print("   - Initial LR: 3e-5")
    print("   - Step size: 10-15 epochs")
    print("   - Gamma: 0.5 (50% 감소)")
    
    print("\n💡 Option 4: Reduce on Plateau (적응적)")
    print("📌 특징:")
    print("   - Validation loss가 개선되지 않으면 lr 감소")
    print("   - 자동으로 최적 시점에 lr 조정")
    print("   - Self-distillation 수렴 특성에 적합")
    print("📌 추천 설정:")
    print("   - Initial LR: 2e-5")
    print("   - Patience: 3-5 epochs")
    print("   - Factor: 0.5")
    print("   - Min LR: 1e-6")

def recommend_best_option():
    """Self-distillation에 가장 적합한 옵션 추천"""
    
    print("\n" + "="*65)
    print("🏆 Self-Distillation Fine-tuning 최고 추천:")
    print("="*65)
    
    print("\n🥇 1순위: Reduce on Plateau")
    print("이유:")
    print("   ✅ Self-distillation은 loss landscape가 복잡함")
    print("   ✅ Teacher-student 수렴 속도가 다를 수 있음")
    print("   ✅ Validation loss 기반 자동 조정이 최적")
    print("   ✅ 과도한 lr 감소 방지")
    
    print("\n🥈 2순위: Constant LR")
    print("이유:")
    print("   ✅ 가장 안전하고 안정적")
    print("   ✅ Fine-tuning의 검증된 방법")
    print("   ✅ Hyperparameter tuning 부담 적음")
    print("   ✅ Reproducible results")
    
    print("\n🥉 3순위: Cosine Annealing")
    print("이유:")
    print("   ✅ 부드러운 수렴")
    print("   ✅ 일반화 성능 향상")
    print("   ⚠️  Total epochs 미리 정해야 함")

def provide_implementation_code():
    """구현 코드 예시 제공"""
    
    print("\n" + "="*65)
    print("💻 구현 코드 (train.py 수정):")
    print("="*65)
    
    print("\n🔧 Option 1: Reduce on Plateau (추천)")
    print("""
# train.py에서 Noam optimizer 대신:
import torch.optim as optim

# Adam optimizer 사용
optimizer = optim.Adam(
    model.parameters(),
    lr=2e-5,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8
)

# ReduceLROnPlateau scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',           # validation loss 기준
    factor=0.5,          # 50% 감소
    patience=3,          # 3 epochs 동안 개선 없으면 감소
    min_lr=1e-6,         # 최소 학습률
    verbose=True         # 로그 출력
)
""")
    
    print("\n🔧 Option 2: Constant LR")
    print("""
# Adam optimizer with constant LR
optimizer = optim.Adam(
    model.parameters(),
    lr=3e-5,             # 고정 학습률
    weight_decay=1e-4
)

# No scheduler needed (또는 dummy scheduler)
scheduler = None
""")
    
    print("\n🔧 Option 3: Cosine Annealing")
    print("""
optimizer = optim.Adam(
    model.parameters(),
    lr=2e-5,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,            # 총 에포크 수
    eta_min=1e-6         # 최소 학습률
)
""")

def suggest_learning_rates():
    """Self-distillation에 적합한 학습률 범위 제안"""
    
    print("\n" + "="*65)
    print("📈 Self-Distillation Fine-tuning 학습률 가이드:")
    print("="*65)
    
    print("\n🎯 추천 학습률 범위:")
    print("📌 Conservative (안전한 시작): 1e-5")
    print("📌 Moderate (표준 설정): 2e-5 ~ 3e-5")
    print("📌 Aggressive (빠른 수렴): 5e-5")
    
    print("\n⚠️  주의사항:")
    print("❌ 1e-4 이상: 너무 높음, 발산 위험")
    print("❌ 1e-6 이하: 너무 낮음, 수렴 너무 느림")
    print("✅ 2e-5: 대부분의 경우에 안정적")
    
    print("\n🔬 실험 순서:")
    print("1️⃣  ReduceLROnPlateau + 2e-5로 시작")
    print("2️⃣  5-10 epochs 후 loss 패턴 관찰")
    print("3️⃣  필요시 초기 lr 조정 (1e-5 or 3e-5)")
    print("4️⃣  Patience, factor 등 세부 조정")

def provide_train_sd_modifications():
    """train_sd.sh 수정사항 제안"""
    
    print("\n" + "="*65)
    print("🛠️  train_sd.sh 수정 방안:")
    print("="*65)
    
    print("\n📝 새로운 파라미터 추가:")
    print("""
# Learning Rate Settings for Fine-tuning
use_noam_scheduler=false      # Noam 대신 다른 스케줄러 사용
optimizer_type="adam"         # "adam" or "adamw"
learning_rate=2e-5           # 고정 학습률 또는 초기 학습률
scheduler_type="plateau"      # "constant", "plateau", "cosine", "step"
scheduler_patience=3          # ReduceLROnPlateau용
scheduler_factor=0.5          # LR 감소 비율
min_learning_rate=1e-6       # 최소 학습률
""")
    
    print("\n📝 train.py 호출부 수정:")
    print("""
python3 ./conformer_ctc_sd/train.py \\
    --use-noam-scheduler $use_noam_scheduler \\
    --optimizer-type $optimizer_type \\
    --learning-rate $learning_rate \\
    --scheduler-type $scheduler_type \\
    --scheduler-patience $scheduler_patience \\
    --scheduler-factor $scheduler_factor \\
    --min-learning-rate $min_learning_rate \\
    # ... 기존 파라미터들
""")

if __name__ == "__main__":
    analyze_scheduler_options()
    recommend_best_option()
    provide_implementation_code()
    suggest_learning_rates()
    provide_train_sd_modifications()
    
    print("\n" + "="*65)
    print("🎯 최종 추천 요약:")
    print("="*65)
    print("1️⃣  ReduceLROnPlateau + Adam (lr=2e-5)")
    print("2️⃣  Patience=3, Factor=0.5, Min_lr=1e-6")
    print("3️⃣  Weight_decay=1e-4")
    print("4️⃣  5-10 epochs 후 성능 모니터링")
    print("5️⃣  필요시 초기 lr을 1e-5 또는 3e-5로 조정")
    print("\n💡 이 설정으로 안정적이고 효과적인 self-distillation이 가능합니다!")
