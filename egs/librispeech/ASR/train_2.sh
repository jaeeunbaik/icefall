#!/bin/bash

# train_sd.sh - LibriSpeech ASR Self-Distillation Training Script
# Usage: bash train_sd.sh

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

set -euo pipefail

# Data Augmentation Controls (modify these as needed)
enable_spec_aug=true          # SpecAugment (frequency/time masking)
enable_musan=true             # MUSAN noise augmentation
enable_cutmix=false 
enable_cutmix=false 
enable_concatenate=false   

# Training parameters
world_size=1 
max_duration=400
valid_max_duration=15         
num_buckets=400               
num_workers=8    
warm_step=10000
lang_dir="./data/lang_bpe_1024"
method="ctc-decoding"

# Model parameters
att_rate=0                    # 0 for pure CTC, >0 for CTC+Attention
num_decoder_layers=0          # 0 for pure CTC

# Other settings
start_epoch=0
master_port=12346
sanity_check=false           # Set to true for OOM checking (slower)
resume_from=/home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/conformer_ctc_sd_proj/libri-light/exp_kl_layer6,12,18_weakaug/models/pretrained_libri-light_6,12,18_weakaug_avg11.pt
valid_interval=5000           # Much larger interval if we enable validation later

# Learning Rate Scheduler Settings (Fine-tuning options)
scheduler_type="plateau"       # "noam", "plateau", "constant"
base_lr=1e-4                 # Base learning rate for plateau/constant schedulers
scheduler_patience=3          # Patience for ReduceLROnPlateau
scheduler_factor=0.5          # Factor for ReduceLROnPlateau (0.5 = 50% reduction)
min_lr=5e-6          

# Validation decoding settings
enable_validation=True
validation_decoding_method="greedy"    # "greedy" or "beam" - use greedy for faster validation
validation_search_beam=10.0            # Beam size for validation (only used if method="beam")
validation_output_beam=5.0             # Output beam for validation (only used if method="beam")
validation_skip_wer=false              # Skip WER computation for even faster validation (디버깅용 - 이제 false로 변경)

# Distillation Hyperparameters
enable_self_distillation=false
distill_layers=3,5,14
distill_loss_type="kl"         # mse, cosine, kl
alpha=0
distill_aggregation=output_avg       # layer_avg: layer 출력을 평균 내고 비교, output_avg: 각 layer loss를 평균
knowledge="attention-map"      # "encoder-output", "attention-map"
distill_temperature=4.0
ema_decay=0.999
ema_start_step=1000
exp_dir=conformer_ctc_sd_proj/finetuning/pretrained_layer6,12,18_weakaug_avg11

#
spec_aug_time_warp_factor=100              # default: 100
spec_aug_num_frame_masks=3                # default: 2  
spec_aug_features_mask_size=27            # default: 27
spec_aug_num_feature_masks=3              # default: 2
spec_aug_frames_mask_size=100             # default: 100
musan_ratio=0.6                           # default: 0.5
snr_range=5,10

#
use_proj_layer=false
proj_layer_training="full-finetuning"       # full-finetuning, only-proj
return_cuts=False
on_the_fly_feats=True
learning_type="asr"

if [ -z "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="/tmp/icefall"
else
    export PYTHONPATH="${PYTHONPATH}:/tmp/icefall"
fi

CUDA_VISIBLE_DEVICES=2 python3 ./conformer_ctc_sd_proj/train.py \
    --exp-dir $exp_dir \
    --master-port $master_port \
    --sanity-check $sanity_check \
    --world-size $world_size \
    --warm-step $warm_step \
    --start-epoch $start_epoch \
    --resume-from $resume_from \
    --att-rate $att_rate \
    --num-decoder-layers $num_decoder_layers \
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
    --enable-validation $enable_validation \
    --valid-interval $valid_interval \
    --validation-decoding-method $validation_decoding_method \
    --validation-search-beam $validation_search_beam \
    --validation-output-beam $validation_output_beam \
    --validation-skip-wer $validation_skip_wer \
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
