CUDA_VISIBLE_DEVICES=0 
python 3_select_neurons.py \
    --data_path /home/wuxinwei/projects/privacy-editing/pn_result/qwen2.5_url.jsonl \
    --text_threshold 0.05 \
    --batch_threshold 0.1 \
    --data_type url \
    --model_name qwen2.5 \
    --sample_num 10  
