python 2_get_neuron_attribution.py \
    --model_path /data2/wuxinwei/model/enron-qwen2.5-1.5B/epoch_0  \
    --data_path /home/wuxinwei/projects/unlearn4llm/get_memorization/qwen1.5b/qwen1.5b_em_memorized_email.txt \
    --num_batch 10 \
    --device 3 \
    --max_length 128 \
    --model_name qwen2.5 \
    --privacy_kind email \
    --use_fp16



