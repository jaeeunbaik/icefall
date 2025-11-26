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

<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=1 python ./conformer_ctc_sd_proj/decode.py \
    --method ctc-decoding \
    --max-duration 20 \
<<<<<<< HEAD
    --epoch 3 \
    --avg 4 \
    --exp-dir conformer_ctc_sd_proj/train70000-epoch77-avg10/exp_kl_layer4,10,16/models \
    --include-proj-layer True \
    --distill-layers 4,10,16
=======
    --epoch 2 \
    --avg 3 \
    --exp-dir conformer_ctc_sd_proj/train70000-epoch77-avg10/exp_kl_layer3,5,16/models \
    --include-proj-layer True \
    --distill-layers 3,5,16
>>>>>>> master
=======
# Step-based decoding (2000-step intervals)
# --step: maximum step to use
# --avg: number of checkpoints to average (from the most recent ones)
CUDA_VISIBLE_DEVICES=3 python ./conformer_ctc_sd_proj/decode.py \
    --method ctc-decoding \
    --max-duration 150 \
    --epoch 2 \
    --avg 3 \
    --exp-dir conformer_ctc_sd_proj/finetuning/hybrid/4layer/exp_4,8,12,16/models \
    --include-proj-layer False \
    --distill-layers 6,12,18

# Epoch-based decoding (legacy, commented out)
# CUDA_VISIBLE_DEVICES=3 python ./conformer_ctc_sd_proj/decode.py \
#     --method ctc-decoding \
#     --max-duration 100 \
#     --epoch 0 \
#     --avg 1 \
#     --exp-dir conformer_ctc_sd_proj/ft_librilight/exp_librilight_checking_ft/models \
#     --include-proj-layer False \
#     --distill-layers 3,5,16
>>>>>>> master
