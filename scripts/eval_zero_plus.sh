# inference + evaluation on zero-shot models
# $1: model code
# $2: device

if [[ $1 = "blip" ]]; then
	CUDA_VISIBLE_DEVICES=$2 python -m src.evaluate \
		--base_model_id Salesforce/blip2-flan-t5-xl \
		--batch_size 8 \
		--max_new_tokens 10 \
		--run_name eval_zero_plus_$1
		
elif [[ $1 = "iblip" ]]; then
	CUDA_VISIBLE_DEVICES=$2 python -m src.evaluate \
		--base_model_id Salesforce/instructblip-vicuna-7b \
		--batch_size 8 \
		--max_new_tokens 10 \
		--run_name eval_zero_plus_$1

elif [[ $1 = "qwen" ]]; then
	CUDA_VISIBLE_DEVICES=$2 python -m src.evaluate \
		--base_model_id Qwen/Qwen2-VL-7B-Instruct \
		--batch_size 8 \
		--max_new_tokens 10 \
		--run_name eval_zero_plus_$1

elif [[ $1 = "llava" ]]; then
	CUDA_VISIBLE_DEVICES=$2 python -m src.evaluate \
		--base_model_id llava-hf/llava-1.5-7b-hf \
		--batch_size 8 \
		--max_new_tokens 10 \
		--run_name eval_zero_plus_$1

elif [[ $1 = "llama" ]]; then
	CUDA_VISIBLE_DEVICES=$2 python -m src.evaluate \
		--base_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
		--batch_size 8 \
		--max_new_tokens 10 \
		--run_name eval_zero_plus_$1
		--split all \
		--data_dir data/v10
fi