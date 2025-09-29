# inference + evaluation on zero-shot models
# $1 model code
# $2 cuda device
# $3 peft id

if [[ $1 = "blip" ]]; then
	CUDA_VISIBLE_DEVICES=$2 python -m src.evaluate \
		--base_model_id Salesforce/blip2-flan-t5-xl \
		--peft_id $3 \
		--batch_size 8 \
		--max_new_tokens 10 \
		--prompt_id QA/none_noneabove \
		--data_dir data/none_plus \
		--run_name eval_ft_plus_$1 \
		# --pilot
		
elif [[ $1 = "iblip" ]]; then
	CUDA_VISIBLE_DEVICES=$2 python -m src.evaluate \
		--base_model_id Salesforce/instructblip-vicuna-7b \
		--peft_id $3 \
		--batch_size 8 \
		--max_new_tokens 10 \
		--prompt_id QA/none_noneabove \
		--data_dir data/none_plus \
		--run_name eval__ft_plus_$1 \
		# --pilot
		
elif [[ $1 = "qwen" ]]; then
	CUDA_VISIBLE_DEVICES=$2 python -m src.evaluate \
		--base_model_id Qwen/Qwen2-VL-7B-Instruct \
		--peft_id $3 \
		--batch_size 4 \
		--max_new_tokens 10 \
		--prompt_id QA/none_noneabove \
		--data_dir data/none_plus \
		--run_name eval_ft_plus_$1 \
		# --pilot
		
elif [[ $1 = "llava" ]]; then
	CUDA_VISIBLE_DEVICES=$2 python -m src.evaluate \
		--base_model_id llava-hf/llava-1.5-7b-hf \
		--peft_id $3 \
		--batch_size 4 \
		--max_new_tokens 10 \
		--prompt_id QA/none_noneabove \
		--data_dir data/none_plus \
		--run_name eval__ft_plus_$1 \
		# --pilot
		
fi


