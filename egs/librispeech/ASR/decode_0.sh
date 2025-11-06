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
    --max-duration 20 \
<<<<<<< HEAD
    --epoch 3 \
    --avg 4 \
    --exp-dir conformer_ctc_sd_proj/train70000-epoch77-avg10/exp_kl_layer9,18/models \
    --include-proj-layer True \
    --distill-layers 9,18
=======
    --epoch 0 \
    --avg 1 \
    --exp-dir /home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/conformer_ctc_sd_proj/finetuning/pretrained_6,12,18_avg11/models \
    --include-proj-layer False \
    --distill-layers 3,5,14
>>>>>>> master
