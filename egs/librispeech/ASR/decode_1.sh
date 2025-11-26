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
<<<<<<< HEAD
    --max-duration 20 \
    --epoch 2 \
    --avg 3 \
    --exp-dir conformer_ctc_sd_proj/train70000-epoch77-avg10/exp_kl_6,12,18/models \
    --include-proj-layer True \
    --distill-layers 6,12,18
=======
    --max-duration 250 \
    --epoch 1 \
    --avg 2 \
    --exp-dir conformer_ctc_sd_proj/self-distillation/kl_6,12,18_alpha0.5_no-musan/models \
=======
    --max-duration 150 \
    --epoch 5 \
    --avg 4 \
    --exp-dir conformer_ctc_sd_proj/finetuning/hybrid/layer_weights/exp_0.7-0.5-0.3/models \
>>>>>>> master
    --include-proj-layer False \
    --distill-layers 3,5,14
>>>>>>> master

