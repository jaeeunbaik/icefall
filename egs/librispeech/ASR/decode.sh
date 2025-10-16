if [ -z "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="/tmp/icefall"
else
    export PYTHONPATH="${PYTHONPATH}:/tmp/icefall"
fi

CUDA_VISIBLE_DEVICES=3 python ./conformer_ctc_sd/decode.py \
    --method ctc-decoding \
    --max-duration 10 \
    --epoch 77 \
    --avg 10 \
    --exp-dir conformer_ctc/exp_clean70000




# if [ -z "${PYTHONPATH:-}" ]; then
#     export PYTHONPATH="/tmp/icefall"
# else
#     export PYTHONPATH="${PYTHONPATH}:/tmp/icefall"
# fi

# CUDA_VISIBLE_DEVICES=3 python ./conformer_ctc_sd/decode.py \
#     --method ctc-decoding \
#     --max-duration 10 \
#     --epoch 6 \
#     --avg 3 \
#     --exp-dir conformer_ctc_sd/exp_att_alpha1_kl_7e-6/models

