# if [ -z "${PYTHONPATH:-}" ]; then
#     export PYTHONPATH="/tmp/icefall"
# else
#     export PYTHONPATH="${PYTHONPATH}:/tmp/icefall"
# fi

# CUDA_VISIBLE_DEVICES=3 python ./conformer_ctc/decode.py \
#     --method ctc-decoding \
#     --max-duration 10 \
#     --epoch 99 \
#     --avg 1




if [ -z "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="/tmp/icefall"
else
    export PYTHONPATH="${PYTHONPATH}:/tmp/icefall"
fi

# Step-based decoding (2000-step intervals)
# --step: maximum step to use
# --avg: number of checkpoints to average (from the most recent ones)
# CUDA_VISIBLE_DEVICES=3 python ./conformer_ctc_sd_proj/decode.py \
#     --method ctc-decoding \
#     --max-duration 100 \
#     --epoch 6 \
#     --avg 5 \
#     --exp-dir conformer_ctc_sd_proj/finetuning/data-aug/exp_specaug-rir/models \
#     --include-proj-layer False

CUDA_VISIBLE_DEVICES=3 python ./conformer_ctc_sd_proj/decode.py \
    --method nbest-rescoring \
    --max-duration 100 \
    --epoch 6 \
    --avg 3 \
    --exp-dir conformer_ctc_sd_proj/finetuning/data-aug/exp_specaug-rir/models \
    --include-proj-layer False \
    --rnn-lm-epoch 9 \
    --rnn-lm-avg 2


# CUDA_VISIBLE_DEVICES=2 python ./conformer_ctc_sd_proj/decode.py \
#     --method rnn-lm \
#     --max-duration 20 \
#     --epoch 6 \
#     --avg 1 \
#     --exp-dir conformer_ctc_sd_proj/finetuning/exp_1126/3layer/exp_6,12,18/models \
#     --include-proj-layer False \
#     --distill-layers 3,5,14 \
#     --rnn-lm-epoch 1 \
#     --rnn-lm-avg 1

# Epoch-based decoding (legacy, commented out)
# CUDA_VISIBLE_DEVICES=3 python ./conformer_ctc_sd_proj/decode.py \
#     --method ctc-decoding \
#     --max-duration 100 \
#     --epoch 0 \
#     --avg 1 \
#     --exp-dir conformer_ctc_sd_proj/ft_librilight/exp_librilight_checking_ft/models \
#     --include-proj-layer False \
#     --distill-layers 3,5,16
