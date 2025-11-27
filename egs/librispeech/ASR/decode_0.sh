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

CUDA_VISIBLE_DEVICES=0 python ./conformer_ctc_sd_proj/decode.py \
    --method ctc-decoding \
    --max-duration 150 \
    --epoch 3 \
    --avg 2 \
    --exp-dir conformer_ctc_sd_proj/finetuning/hybrid/layer_weights/exp_0.3-0.5-0.7/models \
    --include-proj-layer False \
    --distill-layers 3,5,14
