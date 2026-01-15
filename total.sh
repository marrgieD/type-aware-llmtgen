#!/bin/bash

# srun --gres=gpu:nvidia:2 --cpus-per-task=16 --mem=120G python3 train_stage1.py     --train_file ./data/adult-uniform/k1000/train.csv    --output_dir ./checkpoints2/stage1     --num_train_epochs 3     --batch_size 32

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PWD
srun --gres=gpu:nvidia:3 --cpus-per-task=16 --mem=120G python3 ft_opacus.py     --train_file ./data/adult/adult.csv     --validation_file ./data/adult/adult.csv     --output_dir ./checkpoints2/stage2     --model_name_or_path ./checkpoints2/stage1/final     --stage1_checkpoint ./checkpoints2/stage1/final/pytorch_model.bin     --enable_privacy true  --target_epsilon 0.2    --max_grad_norm 1.0     --batch_size 64     --micro_batch_size 4     --lr 1e-4     --num_train_epochs 9     --expert_loss_weight 1.0     --lm_loss_weight 1.0     --lora_rank 16     --save_steps 200  --logging_steps 50 --target_col Income

mv ~/.local/lib/python3.10 ~/.local/lib/python3.10_backup

srun --gres=gpu:nvidia:4 --cpus-per-task=16 --mem=160G --export=ALL,PYTHONNOUSERSITE=1 /data/users/magg13_d_/miniconda3/envs/dp2stage/bin/python -m torch.distributed.run --nproc_per_node=4 --master_port=29500 generate.py --base_model_path ./checkpoints2/stage1/final --lora_path ./checkpoints2/stage2/final --train_data_path ./data/adult/adult.csv --output_file ./synthetic_data_episilon0_restage1.csv --num_samples 48790 --batch_size 32 --target Income

mv ~/.local/lib/python3.10_backup ~/.local/lib/python3.10

# bash  run_full_evaluation.sh