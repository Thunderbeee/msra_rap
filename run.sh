CUDA_VISIBLE_DECIDES=0 torchrun --nproc_per_node 1 --master_port 1054 run.py --dataset multiarith
# CUDA_VISIBLE_DECIDES=0 torchrun --nproc_per_node 1 --master_port 1054 run.py --dataset gsm8k