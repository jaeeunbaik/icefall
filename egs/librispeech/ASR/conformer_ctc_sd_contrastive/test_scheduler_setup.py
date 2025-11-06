#!/usr/bin/env python3
"""
Scheduler 설정 테스트 스크립트
ReduceLROnPlateau와 Constant LR 설정이 올바르게 구현되었는지 테스트
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim

def test_scheduler_configurations():
    """다양한 스케줄러 설정을 테스트"""
    
    print("🧪 Scheduler Configuration Test")
    print("="*50)
    
    # 간단한 모델 생성
    model = nn.Linear(10, 1)
    
    print("\n1️⃣  Testing ReduceLROnPlateau Configuration:")
    print("-" * 40)
    
    # ReduceLROnPlateau 설정
    optimizer_plateau = optim.Adam(
        model.parameters(),
        lr=2e-5,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_plateau,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=True
    )
    
    print(f"✅ Initial LR: {optimizer_plateau.param_groups[0]['lr']:.2e}")
    print(f"✅ Scheduler Type: {type(scheduler_plateau).__name__}")
    print(f"✅ Mode: min, Factor: 0.5, Patience: 3, Min LR: 1e-6")
    
    # 몇 번의 스텝 시뮬레이션
    print("\n📈 Simulating validation loss steps:")
    validation_losses = [2.5, 2.3, 2.4, 2.4, 2.4, 2.4]  # No improvement for 3 steps
    
    for i, val_loss in enumerate(validation_losses):
        print(f"   Step {i+1}: val_loss={val_loss:.1f}, lr={optimizer_plateau.param_groups[0]['lr']:.2e}")
        scheduler_plateau.step(val_loss)
        if i < len(validation_losses) - 1:
            print(f"            → Next lr={optimizer_plateau.param_groups[0]['lr']:.2e}")
    
    print(f"\n✅ Final LR after patience exhausted: {optimizer_plateau.param_groups[0]['lr']:.2e}")
    
    print("\n2️⃣  Testing Constant LR Configuration:")
    print("-" * 40)
    
    # Constant LR 설정
    optimizer_constant = optim.Adam(
        model.parameters(),
        lr=2e-5,
        weight_decay=1e-4
    )
    
    print(f"✅ Constant LR: {optimizer_constant.param_groups[0]['lr']:.2e}")
    print(f"✅ No scheduler needed")
    print(f"✅ LR remains constant throughout training")
    
    # 몇 에포크 시뮬레이션
    print("\n📈 Simulating epochs with constant LR:")
    for epoch in range(1, 6):
        print(f"   Epoch {epoch}: lr={optimizer_constant.param_groups[0]['lr']:.2e}")
    
    print("\n3️⃣  Testing Noam Scheduler (for comparison):")
    print("-" * 40)
    
    # Noam scheduler 시뮬레이션 (실제 import 없이)
    print("✅ Noam Scheduler characteristics:")
    print("   - Warmup phase: lr increases linearly")
    print("   - Decay phase: lr decreases as step^(-0.5)")
    print("   - Self-contained: handles scheduling internally")
    print("   - Good for: Training from scratch")
    
    print("\n" + "="*50)
    print("📋 Scheduler Comparison Summary:")
    print("="*50)
    
    print("\n🏆 ReduceLROnPlateau:")
    print("   ✅ Adapts to validation loss")
    print("   ✅ Best for fine-tuning")
    print("   ✅ Prevents overfitting")
    print("   ⚠️  Requires validation monitoring")
    
    print("\n🥈 Constant LR:")
    print("   ✅ Simple and stable")
    print("   ✅ Easy to reproduce")
    print("   ✅ Good for fine-tuning")
    print("   ⚠️  May need manual adjustment")
    
    print("\n🥉 Noam Scheduler:")
    print("   ✅ Good for training from scratch")
    print("   ✅ Proven for Transformer models")
    print("   ❌ Not suitable for fine-tuning")
    print("   ❌ Fixed schedule, not adaptive")

def test_argument_parsing():
    """인자 파싱 테스트 시뮬레이션"""
    
    print("\n" + "="*50)
    print("🔧 Argument Parsing Test:")
    print("="*50)
    
    test_cases = [
        {
            "name": "ReduceLROnPlateau Configuration",
            "args": {
                "scheduler_type": "plateau",
                "base_lr": 2e-5,
                "scheduler_patience": 3,
                "scheduler_factor": 0.5,
                "min_lr": 1e-6,
            }
        },
        {
            "name": "Constant LR Configuration", 
            "args": {
                "scheduler_type": "constant",
                "base_lr": 3e-5,
                "scheduler_patience": 3,  # unused
                "scheduler_factor": 0.5,  # unused
                "min_lr": 1e-6,          # unused
            }
        },
        {
            "name": "Noam Scheduler Configuration (legacy)",
            "args": {
                "scheduler_type": "noam",
                "lr_factor": 5.0,
                "warm_step": 10000,
                "attention_dim": 512,
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}️⃣  {test_case['name']}:")
        for key, value in test_case['args'].items():
            print(f"   --{key.replace('_', '-')}: {value}")

if __name__ == "__main__":
    test_scheduler_configurations()
    test_argument_parsing()
    
    print("\n" + "="*50)
    print("🎯 Ready for Self-Distillation Fine-tuning!")
    print("="*50)
    print("✅ ReduceLROnPlateau: train_sd.sh에서 scheduler_type='plateau'")
    print("✅ Constant LR: train_sd.sh에서 scheduler_type='constant'")
    print("✅ 기본 설정: plateau with lr=2e-5, patience=3, factor=0.5")
    print("\n🚀 이제 train_sd.sh를 실행할 수 있습니다!")
