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

CUDA_VISIBLE_DEVICES=3 python ./conformer_ctc_sd_proj/decode.py \
    --method ctc-decoding \
    --max-duration 10 \
<<<<<<< HEAD
    --epoch 1 \
    --avg 1 \
    --exp-dir conformer_ctc_sd_proj/train70000-epoch77-avg10/exp_mse_1e-5_0.1:1/models \
    --include-proj-layer False \
=======
    --epoch 9 \
    --avg 4 \
    --exp-dir conformer_ctc_sd_proj/train70000-epoch77-avg10/exp_mse_1e-5_0.1:1/models \
    --include-proj-layer True \
>>>>>>> 28958b141bb31ecff6a3230378a4b2d4c3046072
    --distill-layers 17
