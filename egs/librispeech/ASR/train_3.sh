#!/bin/bash

# train_sd.sh - LibriSpeech ASR Self-Distillation Training Script
# Usage: bash train_sd.sh

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

set -euo pipefail

world_size=1 
master_port=12346
num_epochs=10
start_epoch=0
exp_dir=conformer_ctc_sd_proj/self-distillation/kl_6,12,18_alpha0.5_no-musan
lang_dir="./data/lang_bpe_1024"
sanity_check=false           # Set to true for OOM checking (slower)


# Learning Rate Scheduler Settings (Fine-tuning options)
scheduler_type="plateau"       # "noam", "plateau", "constant"
base_lr=3e-5                 # Base learning rate for plateau/constant schedulers
scheduler_patience=3          # Patience for ReduceLROnPlateau
scheduler_factor=0.5          # Factor for ReduceLROnPlateau (0.5 = 50% reduction)
min_lr=5e-6          


# Distillation Hyperparameters
enable_self_distillation=true
distill_layers=6,12,18
distill_loss_type="kl"         # mse, cosine, kl
alpha=1000.0
distill_aggregation=output_avg       # layer_avg: layer 출력을 평균 내고 비교, output_avg: 각 layer loss를 평균
distill_temperature=4.0
layer_weights=2.0,3.0,4.0
ema_decay=0.999
ema_start_step=1000

# Training Schema
learning_type="hybrid"
use_proj_layer=true


# Data Augmentation Controls (modify these as needed)
enable_spec_aug=true          # SpecAugment (frequency/time masking)
enable_musan=false             # MUSAN noise augmentation
enable_cutmix=false 
enable_concatenate=false   

# Training parameters

max_duration=220
valid_max_duration=15         
num_buckets=220               
num_workers=8    
warm_step=10000
method="ctc-decoding"


# Other settings
resume_from=/home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/zoo/conformer_ctc_70000_from77avg10.pt
valid_interval=5000           # Much larger interval if we enable validation later

#
spec_aug_time_warp_factor=100              # default: 100
spec_aug_num_frame_masks=6                # default: 2  
spec_aug_features_mask_size=27            # default: 27
spec_aug_num_feature_masks=6              # default: 2
spec_aug_frames_mask_size=100             # default: 100
musan_ratio=0.9                           # default: 0.5
snr_range=5,10

#
return_cuts=False
on_the_fly_feats=True
# Prototype 최적화 설정 추가
prototype_dir="./prototypes/librispeech_clean_256"  # 명확한 경로
num_prototypes=256              # 적절한 크기
prototype_samples=50000         # 샘플 수 줄여서 빠른 초기화


if [ -z "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="/tmp/icefall"
else
    export PYTHONPATH="${PYTHONPATH}:/tmp/icefall"
fi

CUDA_VISIBLE_DEVICES=3 python3 ./conformer_ctc_sd_proj/train.py \
    --exp-dir $exp_dir \
    --master-port $master_port \
    --sanity-check $sanity_check \
    --world-size $world_size \
    --warm-step $warm_step \
    --start-epoch $start_epoch \
    --resume-from $resume_from \
    --num-workers $num_workers \
    --enable-spec-aug $enable_spec_aug \
    --enable-musan $enable_musan \
    --max-duration $max_duration \
    --valid-max-duration $valid_max_duration \
    --num-buckets $num_buckets \
    --bucketing-sampler false \
    --concatenate-cuts $enable_concatenate \
    --duration-factor 1.0 \
    --drop-last true \
    --shuffle true \
    --lang-dir $lang_dir \
    --method $method \
    --scheduler-type $scheduler_type \
    --base-lr $base_lr \
    --scheduler-patience $scheduler_patience \
    --scheduler-factor $scheduler_factor \
    --min-lr $min_lr \
    --enable-self-distillation $enable_self_distillation \
    --distill-layers $distill_layers \
    --distill-loss-type $distill_loss_type \
    --alpha $alpha \
    --distill-aggregation $distill_aggregation \
    --distill-temperature $distill_temperature \
    --ema-decay $ema_decay \
    --ema-start-step $ema_start_step \
    --use-proj-layer $use_proj_layer \
    --return-cuts $return_cuts \
    --on-the-fly-feats $on_the_fly_feats \
    --learning-type $learning_type \
    --prototype-dir $prototype_dir \
    --num-prototypes $num_prototypes \
    --prototype-samples $prototype_samples