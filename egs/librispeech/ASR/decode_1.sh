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
    --max-duration 20 \
    --epoch 1 \
    --avg 2 \
    --exp-dir conformer_ctc_sd_proj/train70000-epoch77-avg10/exp_mse_1e-5_10:1/models \
    --include-proj-layer False \
    --distill-layers 17

