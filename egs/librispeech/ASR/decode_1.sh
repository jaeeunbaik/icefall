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
<<<<<<< HEAD
    --max-duration 20 \
    --epoch 2 \
    --avg 3 \
    --exp-dir conformer_ctc_sd_proj/train70000-epoch77-avg10/exp_ft_high-clean_normal-aug/models \
    --include-proj-layer False \
=======
    --max-duration 250 \
    --epoch 1 \
    --avg 2 \
    --exp-dir conformer_ctc_sd_proj/self-distillation/kl_6,12,18_alpha0.5_no-musan/models \
    --include-proj-layer False \
    --distill-layers 3,5,14
>>>>>>> master

