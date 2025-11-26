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

CUDA_VISIBLE_DEVICES=2 python ./conformer_ctc_sd_proj/decode.py \
    --method ctc-decoding \
<<<<<<< HEAD
    --max-duration 20 \
    --epoch 2 \
    --avg 3 \
<<<<<<< HEAD
    --exp-dir conformer_ctc_sd_proj/train70000-epoch77-avg10/exp_ft_high-clean_strong-aug/models \
    --include-proj-layer False \
=======
    --exp-dir conformer_ctc_sd_proj/train70000-epoch77-avg10/exp_kl_layer14/models \
    --include-proj-layer True \
    --distill-layers 14
>>>>>>> master
=======
    --max-duration 150 \
    --epoch 5 \
    --avg 4 \
    --exp-dir conformer_ctc_sd_proj/finetuning/hybrid/layer_weights/exp_0.5-0.5-1.0/models \
    --include-proj-layer False \
    --distill-layers 14

# CUDA_VISIBLE_DEVICES=2 python ./conformer_ctc_sd_proj/decode.py \
#     --method ctc-decoding \
#     --max-duration 150 \
#     --step 30000 \
#     --avg 6 \
#     --exp-dir conformer_ctc_sd_proj/finetuning/3layer_1120/exp_4,8,12/models \
#     --include-proj-layer False \
#     --distill-layers 14
>>>>>>> master
