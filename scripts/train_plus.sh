# $1: model id
# $2: cuda device

if [ $1 == "qwen" ]; then
    CUDA_VISIBLE_DEVICES=$2 python -m src.train_pt \
		--model_name Qwen/Qwen2-VL-7B-Instruct \
		--num_epochs 3 \
		--max_new_tokens 10 \
		--train_batch_size 1 \
		--eval_batch_size 4 \
		--eval_freq 0.2 \
		--logging_step 100 \
        --data_dir data/none_plus \
		--output_dir weights/train_plus_qwen \
		--lr 1e-5
elif [ $1 == "blip" ]; then
    CUDA_VISIBLE_DEVICES=$2 python -m src.train_pt \
        --logging_step 100 \
        --eval_freq 0.2 \
        --model_name Salesforce/blip2-flan-t5-xl \
        --num_epochs 3 \
        --max_new_tokens 10 \
        --train_batch_size 4 \
        --eval_batch_size 4 \
		--data_dir data/none_plus \
        --output_dir weights/train_plus_blip \
        --lr 1e-5
elif [ $1 == "iblip" ]; then
    CUDA_VISIBLE_DEVICES=$2 python -m src.train_pt \
		--model_name Salesforce/instructblip-vicuna-7b \
		--num_epochs 3 \
		--max_new_tokens 10 \
		--train_batch_size 4 \
		--eval_batch_size 4 \
		--eval_freq 0.2 \
		--logging_step 100 \
		--data_dir data/none_plus \
		--output_dir weights/train_plus_iblip \
		--lr 1e-5
elif [ $1 == "llava" ]; then
    CUDA_VISIBLE_DEVICES=$2 python -m src.train_pt \
        --model_name llava-hf/llava-1.5-7b-hf \
        --num_epochs 3 \
        --max_new_tokens 10 \
        --train_batch_size 2 \
        --eval_batch_size 8 \
        --eval_freq 0.2 \
        --logging_step 100 \
		--data_dir data/none_plus \
        --output_dir weights/train_plus_llava \
    	--lr 1e-5
else
    echo "Invalid model id: $1"
    exit 1
fi