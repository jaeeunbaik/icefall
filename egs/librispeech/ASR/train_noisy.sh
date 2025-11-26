#!/bin/# Data Augmentation Controls (modify these as needed)

set -euo pipefail

# Training parameters
world_size=4                    # Multi-GPU restored since test passed
max_duration=430                # 4x increase for GPU memory utilization (100->400)
valid_max_duration=50          # 4x increase to match (15->60)
num_buckets=300                 # 2x increase for better bucketing with larger batches
num_workers=16                  # Keep same for stability
lang_dir="./data/lang_bpe_1024"
exp_dir="conformer_ctc/exp_pretrain_random_init_recluster-2_tw80_6,12,18_ft_10000_1118"   
method="ctc-decoding"
warm_step=10000         
lr_factor=5.0               
weight_decay=1e-6          

# Model parameters
att_rate=0                    # 0 for pure CTC, >0 for CTC+Attention
num_decoder_layers=0          # 0 for pure CTC

# Augmentation
enable_spec_aug=True
enable_musan=True

# Other settings
start_epoch=0
master_port=12345
sanity_check=false           # Set to true for OOM checking (slower)

# Validation settings - more frequent validation to catch improvements early
valid_interval=50000           # More frequent validation to catch improvements

# Validation decoding settings
validation_decoding_method="greedy"    # "greedy" or "beam" - use greedy for faster validation
validation_search_beam=20.0            # Beam size for validation (only used if method="beam")
validation_output_beam=8.0             # Output beam for validation (only used if method="beam")
validation_skip_wer=false              # Skip WER computation for even faster validation (디버깅용 - 이제 false로 변경)

init_model_from_pretrain=/home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/zoo/pretrain_random_init_recluster-2_tw80_6,12,18_avg5.pt
# gdb --args python ./conformer_ctc/train.py
if [ -z "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="/tmp/icefall"
else
    export PYTHONPATH="${PYTHONPATH}:/tmp/icefall"
fi


python3 ./conformer_ctc/train.py \
    --master-port $master_port \
    --world-size $world_size \
    --warm-step $warm_step \
    --lr-factor $lr_factor \
    --weight-decay $weight_decay \
    --enable-spec-aug $enable_spec_aug \
    --enable-musan $enable_musan \
    --start-epoch $start_epoch \
    --att-rate $att_rate \
    --num-decoder-layers $num_decoder_layers \
    --num-workers $num_workers \
    --on-the-fly-feats false \
    --max-duration $max_duration \
    --valid-max-duration $valid_max_duration \
    --num-buckets $num_buckets \
    --bucketing-sampler true \
    --duration-factor 1.0 \
    --drop-last true \
    --shuffle true \
    --lang-dir $lang_dir \
    --exp-dir $exp_dir \
    --init-model-from-pretrain $init_model_from_pretrain \
    --debug-train false \
    --debug-first-n-batches 5
