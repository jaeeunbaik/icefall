
#!/bin/bash

# train_sd.sh - LibriSpeech ASR Self-Distillation Training Script
# Usage: bash train_sd.sh

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

set -euo pipefail

# Training parameters
world_size=1 
max_duration=300
valid_max_duration=15         
num_buckets=300               
num_workers=6    
lang_dir="./data/lang_bpe_1024"
method="ctc-decoding"

# Model parameters
att_rate=0                    # 0 for pure CTC, >0 for CTC+Attention
num_decoder_layers=0          # 0 for pure CTC

# Other settings
start_epoch=0
master_port=12346
sanity_check=false           # Set to true for OOM checking (slower)
resume_from=/home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/conformer_ctc_sd_proj/libri-light/layer_weights/exp_0.5-0.5-1.0/models/averaged_10-20000.pt
enable_validation=true       # Temporarily disable validation to avoid crashes
valid_interval=10000           # Much larger interval if we enable validation later

# Learning Rate Scheduler Settings (Fine-tuning options)
scheduler_type="noam"       # "noam", "plateau", "constant"
lr_factor=0.1                 # lr_factor for Noam scheduler, fine-tuning often needs smaller values
warm_step=10000                # Warmup steps for Noam, reduced for fine-tuning
# base_lr=5e-4                 # Base learning rate for plateau/constant schedulers
# scheduler_patience=2          # Patience for ReduceLROnPlateau
# scheduler_factor=0.5          # Factor for ReduceLROnPlateau (0.5 = 50% reduction)
# min_lr=5e-6          

# Validation decoding settings
validation_decoding_method="greedy"    # "greedy" or "beam" - use greedy for faster validation
validation_search_beam=10.0            # Beam size for validation (only used if method="beam")
validation_output_beam=5.0             # Output beam for validation (only used if method="beam")
validation_skip_wer=false              # Skip WER computation for even faster validation (디버깅용 - 이제 false로 변경)

# Distillation Hyperparameters
enable_self_distillation=true
distill_layers=5,11,17
distill_loss_type="kl"         # mse, cosine, kl
alpha=1000
distill_aggregation=output_avg       # layer_avg: layer 출력을 평균 내고 비교, output_avg: 각 layer loss를 평균
distill_temperature=4.0
ema_decay=0.999
ema_start_step=1000
exp_dir=conformer_ctc_sd_proj/finetuning/hybrid/layer_weights/exp_0.5-0.5-1.0



# Data Augmentation Controls (modify these as needed)
clean_enable_spec_aug=false          # SpecAugment (frequency/time masking)
clean_enable_musan=false             # MUSAN noise augmentation
clean_enable_cutmix=false 
clean_enable_concatenate=false
clean_enable_rir=false

clean_spec_aug_time_warp_factor=80              # default: 100
clean_spec_aug_num_frame_masks=2                # default: 2  
clean_spec_aug_features_mask_size=27            # default: 27
clean_spec_aug_num_feature_masks=10              # default: 2
clean_spec_aug_frames_mask_size=100             # default: 100
clean_musan_ratio=0.5                           # default: 0.5
clean_snr_range=10,20
clean_rir_prob=0.5



# Data Augmentation Controls (modify these as needed)
noisy_enable_spec_aug=true          # SpecAugment (frequency/time masking)
noisy_enable_musan=false             # MUSAN noise augmentation
noisy_enable_cutmix=false 
noisy_enable_concatenate=false
noisy_enable_rir=false

noisy_spec_aug_time_warp_factor=80              # default: 100
noisy_spec_aug_num_frame_masks=2                # default: 2  
noisy_spec_aug_features_mask_size=27            # default: 27
noisy_spec_aug_num_feature_masks=2              # default: 2
noisy_spec_aug_frames_mask_size=100             # default: 100
noisy_musan_ratio=0.5                           # default: 0.5
noisy_snr_range=10,20
noisy_rir_prob=0.5
return_cuts=False
on_the_fly_feats=false


#
use_proj_layer=true
return_cuts=True
on_the_fly_feats=False
learning_type="hybrid"


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
    --max-duration $max_duration \
    --valid-max-duration $valid_max_duration \
    --num-buckets $num_buckets \
    --bucketing-sampler true \
    --duration-factor 1.0 \
    --drop-last true \
    --shuffle false \
    --lang-dir $lang_dir \
    --method $method \
    --scheduler-type $scheduler_type \
    --lr-factor $lr_factor \
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
    --clean-enable-spec-aug $clean_enable_spec_aug \
    --clean-enable-musan $clean_enable_musan \
    --clean-enable-cutmix $clean_enable_cutmix \
    --clean-enable-concatenate $clean_enable_concatenate \
    --clean-enable-rir $clean_enable_rir \
    --clean-spec-aug-time-warp-factor $clean_spec_aug_time_warp_factor \
    --clean-spec-aug-num-frame-masks $clean_spec_aug_num_frame_masks \
    --clean-spec-aug-features-mask-size $clean_spec_aug_features_mask_size \
    --clean-spec-aug-num-feature-masks $clean_spec_aug_num_feature_masks \
    --clean-spec-aug-frames-mask-size $clean_spec_aug_frames_mask_size \
    --clean-musan-ratio $clean_musan_ratio \
    --clean-snr-range $clean_snr_range \
    --clean-rir-prob $clean_rir_prob \
    --noisy-enable-spec-aug $noisy_enable_spec_aug \
    --noisy-enable-musan $noisy_enable_musan \
    --noisy-enable-cutmix $noisy_enable_cutmix \
    --noisy-enable-concatenate $noisy_enable_concatenate \
    --noisy-enable-rir $noisy_enable_rir \
    --noisy-spec-aug-time-warp-factor $noisy_spec_aug_time_warp_factor \
    --noisy-spec-aug-num-frame-masks $noisy_spec_aug_num_frame_masks \
    --noisy-spec-aug-features-mask-size $noisy_spec_aug_features_mask_size \
    --noisy-spec-aug-num-feature-masks $noisy_spec_aug_num_feature_masks \
    --noisy-spec-aug-frames-mask-size $noisy_spec_aug_frames_mask_size \
    --noisy-musan-ratio $noisy_musan_ratio \
    --noisy-snr-range $noisy_snr_range \
    --noisy-rir-prob $noisy_rir_prob