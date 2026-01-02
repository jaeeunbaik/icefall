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



'''
GREEDY DECODING
'''
CUDA_VISIBLE_DEVICES=0 python ./conformer_ctc_sd_proj/decode.py \
    --method ctc-decoding \
    --max-duration 100 \
    --epoch 8 \
    --avg 4 \
    --exp-dir conformer_ctc_sd_proj/finetuning/exp_1126/3layer/exp_6,12,18/models \
    --include-proj-layer False \
    --distill-layers 3,5,14


"""
N-GRAM DECODING
"""
# CUDA_VISIBLE_DEVICES=1 python ./conformer_ctc_sd_proj/decode.py \
#     --method nbest-rescoring \
#     --max-duration 20 \
#     --epoch 6 \
#     --avg 1 \
#     --exp-dir conformer_ctc_sd_proj/finetuning/exp_1126/3layer/exp_6,12,18/models \
#     --include-proj-layer False \
#     --distill-layers 3,5,14



"""
RNN-LM DECODING
"""
# CUDA_VISIBLE_DEVICES=2 python ./conformer_ctc_sd_proj/decode.py \
#     --method rnn-lm \
#     --max-duration 20 \
#     --epoch 6 \
#     --avg 1 \
#     --exp-dir conformer_ctc_sd_proj/finetuning/exp_1126/3layer/exp_6,12,18/models \
#     --include-proj-layer False \
#     --distill-layers 3,5,14 \
#     --rnn-lm-epoch 1 \
#     --rnn-lm-avg 1