
set -euo pipefail

# Training parameters
world_size=4                    # Multi-GPU restored since test passed
max_duration=200                # 4x increase for GPU memory utilization (100->400)
valid_max_duration=20          # 4x increase to match (15->60)
num_buckets=200                 # 2x increase for better bucketing with larger batches
num_workers=8                  # Keep same for stability
lang_dir="./data/lang_bpe_1024"
exp_dir="conformer_ctc_siam/exp_siam_70000"    # Updated for new warmup steps
method="ctc-decoding"
warm_step=70000                # Reduced to ~20% of total steps (more reasonable)
lr_factor=5.0                  # Even smaller learning rate factor
weight_decay=1e-6              # Stronger regularization
siamese_loss_weight=0.4

# Model parameters - HYBRID CONFIGURATION
att_rate=0.8                  # 30% Attention + 70% CTC for balanced hybrid training
num_decoder_layers=6          # 6 transformer decoder layers for hybrid mode

# =================================================================
# NOISE AUGMENTATION PARAMETERS
# =================================================================

# SpecAugment: Frequency and time masking augmentation
enable_spec_aug=True

# MUSAN: Music, Speech and Noise corpus for noise augmentation  
enable_musan=True                 # Enable MUSAN noise augmentation for noisy data
musan_ratio=0.9                   # Probability of applying MUSAN (80% of samples)
musan_snr_range="0,5"          # SNR range for MUSAN noise (10-20 dB)

# RIR: Room Impulse Response augmentation (simulates room acoustics)
enable_rir=False                  # Room impulse response augmentation
rir_ratio=0.5                     # Probability of applying RIR

# Cut Concatenation: Concatenate multiple cuts for longer sequences
# WARNING: Disabled for Siamese training as it causes length mismatch between clean/noisy pairs
enable_concatenate=False          # DISABLED for Siamese: causes sequence length mismatch


# =================================================================
# NOTE: In Siamese training mode:
# - Clean dataset: No augmentations applied (clean reference)
# - Noisy dataset: All enabled augmentations applied
# - Concatenation is DISABLED: Prevents sequence length mismatch
#   between clean/noisy pairs which would break consistency loss
# - Only length-preserving augmentations are safe: MUSAN, RIR, SpecAugment
# =================================================================

# Other settings
start_epoch=0                 # Start from scratch for hybrid training
master_port=12346             # Different port to avoid conflicts
sanity_check=false           # Set to true for OOM checking (slower)

# Validation settings - more frequent validation to catch improvements early
valid_interval=1000           # Very frequent validation to monitor convergence (was 5000)

# Validation decoding settings
validation_decoding_method="greedy"    # "greedy" or "beam" - use greedy for faster validation
validation_search_beam=20.0            # Beam size for validation (only used if method="beam")
validation_output_beam=8.0             # Output beam for validation (only used if method="beam")
validation_skip_wer=false              # Skip WER computation for even faster validation (디버깅용 - 이제 false로 변경)


if [ -z "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="/tmp/icefall"
else
    export PYTHONPATH="${PYTHONPATH}:/tmp/icefall"
fi


python ./conformer_ctc_siam/train.py \
    --enable-siamese true \
    --att-rate 0.5 \
    --num-decoder-layers 6 \
    --siamese-loss-weight $siamese_loss_weight \
    --siamese-warmup-steps 8000 \
    --master-port $master_port \
    --world-size $world_size \
    --warm-step $warm_step \
    --lr-factor $lr_factor \
    --weight-decay $weight_decay \
    --enable-spec-aug $enable_spec_aug \
    --enable-musan $enable_musan \
    --musan-ratio $musan_ratio \
    --snr-range $musan_snr_range \
    --start-epoch $start_epoch \
    --att-rate $att_rate \
    --num-decoder-layers $num_decoder_layers \
    --num-workers $num_workers \
    --on-the-fly-feats true \
    --max-duration $max_duration \
    --valid-max-duration $valid_max_duration \
    --num-buckets $num_buckets \
    --bucketing-sampler true \
    --duration-factor 1.0 \
    --drop-last true \
    --shuffle true \
    --lang-dir $lang_dir \
    --exp-dir $exp_dir \
    --debug-train false \
    --debug-first-n-batches 5 