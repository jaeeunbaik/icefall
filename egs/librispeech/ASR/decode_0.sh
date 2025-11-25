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
    --epoch 2 \
    --avg 3 \
    --exp-dir conformer_ctc_sd_proj/finetuning/layer_weights/hybrid/exp_0.3-0.5-0.7/models \
    --include-proj-layer False \
    --distill-layers 3,5,14
