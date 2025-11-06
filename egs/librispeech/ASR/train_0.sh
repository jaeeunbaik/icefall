#!/bin/bash

# train_sd.sh - LibriSpeech ASR Self-Distillation Training Script
# Usage: bash train_sd.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

set -euo pipefail

# Data Augmentation Controls (modify these as needed)
enable_spec_aug=true          # SpecAugment (frequency/time masking)
enable_musan=true             # MUSAN noise augmentation
enable_concatenate=false   

# Training parameters
world_size=1 
max_duration=400
valid_max_duration=15         
num_buckets=400               
num_workers=8    
warm_step=10000
lang_dir="/home/nas4/user/jaeeun/icefall/egs/librispeech/ASR/data/lang_bpe_1024/"
manifest_dir="/home/nas4/user/jaeeun/icefall/egs/librispeech/ASR/data/fbank/"
method="ctc-decoding"

# Model parameters
att_rate=0                    # 0 for pure CTC, >0 for CTC+Attention
num_decoder_layers=0          # 0 for pure CTC

# Other settings
start_epoch=8
master_port=12346
sanity_check=false           # Set to true for OOM checking (slower)
<<<<<<< HEAD
resume_from=/home/nas4/user/jaeeun/icefall/egs/librispeech/ASR/zoo/conformer_ctc_70000_from77avg10.pt
=======
resume_from=/home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/conformer_ctc_sd_proj/libri-light/exp_kl_layer6,12,18/models/pretrained_libri-light_6,12,18_average11.pt
>>>>>>> master
enable_validation=true       # Temporarily disable validation to avoid crashes
valid_interval=500000           # Much larger interval if we enable validation later

# Learning Rate Scheduler Settings (Fine-tuning options)
scheduler_type="plateau"       # "noam", "plateau", "constant"
base_lr=1e-4                 # Base learning rate for plateau/constant schedulers
scheduler_patience=3          # Patience for ReduceLROnPlateau
scheduler_factor=0.5          # Factor for ReduceLROnPlateau (0.5 = 50% reduction)
min_lr=5e-6            

# Validation decoding settings
validation_decoding_method="greedy"    # "greedy" or "beam" - use greedy for faster validation
validation_search_beam=10.0            # Beam size for validation (only used if method="beam")
validation_output_beam=5.0             # Output beam for validation (only used if method="beam")
validation_skip_wer=false              # Skip WER computation for even faster validation (디버깅용 - 이제 false로 변경)

# Distillation Hyperparameters
<<<<<<< HEAD
enable_self_distillation=true
distill_layers=5,11,17
distill_loss_type="kl"         # mse, cosine, kl
alpha=500000000000
=======
enable_self_distillation=false
distill_layers=3,5,14
distill_loss_type="kl"         # mse, cosine, kl
alpha=0
>>>>>>> master
distill_aggregation=output_avg       # layer_avg: layer 출력을 평균 내고 비교, output_avg: 각 layer loss를 평균
distill_temperature=4.0
ema_decay=0.999
ema_start_step=1000
<<<<<<< HEAD
clean_ratio=0.1
exp_dir=conformer_ctc_sd_proj/train70000-epoch77-avg10/exp_kl_layer5,11,17_strong-aug

#
spec_aug_time_warp_factor=0              # default: 100
spec_aug_num_frame_masks=4                # default: 2  
spec_aug_features_mask_size=27            # default: 27
spec_aug_num_feature_masks=4              # default: 2
spec_aug_frames_mask_size=100             # default: 100
musan_ratio=0.8                           # default: 0.5
snr_range=0,5

#
use_proj_layer=True
proj_layer_training="full-finetuning"       # full-finetuning, only-proj
=======
exp_dir=conformer_ctc_sd_proj/finetuning/pretrained_6,12,18_avg11

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
>>>>>>> master
return_cuts=False
on_the_fly_feats=True
learning_type="asr"

if [ -z "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="/tmp/icefall"
else
    export PYTHONPATH="${PYTHONPATH}:/tmp/icefall"
fi

CUDA_VISIBLE_DEVICES=0 python3 ./conformer_ctc_sd_proj/train.py \
    --enable-spec-aug $enable_spec_aug \
    --enable-musan $enable_musan \
    --enable-concatenate $enable_concatenate \
    --world-size $world_size \
    --max-duration $max_duration \
    --valid-max-duration $valid_max_duration \
    --num-buckets $num_buckets \
    --num-workers $num_workers \
    --warm-step $warm_step \
    --lang-dir $lang_dir \
    --method $method \
    --att-rate $att_rate \
    --num-decoder-layers $num_decoder_layers \
    --start-epoch $start_epoch \
    --master-port $master_port \
    --sanity-check $sanity_check \
    --resume-from $resume_from \
    --enable-validation $enable_validation \
    --valid-interval $valid_interval \
    --scheduler-type $scheduler_type \
    --base-lr $base_lr \
    --scheduler-patience $scheduler_patience \
    --scheduler-factor $scheduler_factor \
    --min-lr $min_lr \
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
    --exp-dir $exp_dir \
    --spec-aug-time-warp-factor $spec_aug_time_warp_factor \
    --spec-aug-num-frame-masks $spec_aug_num_frame_masks \
    --spec-aug-features-mask-size $spec_aug_features_mask_size \
    --spec-aug-num-feature-masks $spec_aug_num_feature_masks \
    --spec-aug-frames-mask-size $spec_aug_frames_mask_size \
    --musan-ratio $musan_ratio \
    --snr-range $snr_range \
    --use-proj-layer $use_proj_layer \
    --return-cuts $return_cuts \
    --on-the-fly-feats $on_the_fly_feats \
<<<<<<< HEAD
    --bucketing-sampler false \
    --duration-factor 1.0 \
    --drop-last true \
    --shuffle true \
    
=======
    --learning-type $learning_type \
>>>>>>> master
