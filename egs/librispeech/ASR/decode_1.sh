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
    --max-duration 200 \
    --epoch 5 \
    --avg 6 \
    --exp-dir conformer_ctc_sd_proj/finetuning/4layer/exp_4,6,12,18/models \
    --include-proj-layer False \
    --distill-layers 3,5,14

