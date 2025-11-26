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
<<<<<<< HEAD
    --max-duration 20 \
<<<<<<< HEAD
    --epoch 2 \
    --avg 3 \
    --exp-dir conformer_ctc_sd_proj/train70000-epoch77-avg10/exp_ft_normal-aug/models \
    --include-proj-layer False
=======
    --epoch 0 \
    --avg 1 \
    --exp-dir /home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/conformer_ctc_sd_proj/finetuning/pretrained_6,12,18_avg11/models \
=======
    --max-duration 150 \
    --epoch 2 \
    --avg 3 \
    --exp-dir conformer_ctc_sd_proj/finetuning/layer_weights/hybrid/exp_0.3-0.5-0.7/models \
>>>>>>> master
    --include-proj-layer False \
    --distill-layers 3,5,14
>>>>>>> master
