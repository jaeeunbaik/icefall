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

CUDA_VISIBLE_DEVICES=1 python ./conformer_ctc_sd_proj/decode.py \
    --method ctc-decoding \
    --max-duration 250 \
    --epoch 1 \
    --avg 2 \
    --exp-dir conformer_ctc_sd_proj/self-distillation/kl_6,12,18_alpha0.5_no-musan/models \
    --include-proj-layer False \
    --distill-layers 3,5,14

