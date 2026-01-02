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

CUDA_VISIBLE_DEVICES=2 python ./conformer_ctc_sd_proj/decode.py \
    --method ctc-decoding \
    --max-duration 100 \
    --epoch 12 \
    --avg 4 \
    --exp-dir conformer_ctc_sd_proj/finetuning/data-aug/exp_musan-rir/models \
    --include-proj-layer False \
    --distill-layers 14

# CUDA_VISIBLE_DEVICES=2 python ./conformer_ctc_sd_proj/decode.py \
#     --method ctc-decoding \
#     --max-duration 150 \
#     --step 30000 \
#     --avg 6 \
#     --exp-dir conformer_ctc_sd_proj/finetuning/3layer_1120/exp_4,8,12/models \
#     --include-proj-layer False \
#     --distill-layers 14
