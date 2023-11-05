CUDA_VISIBLE_DECIDES=0 torchrun --nproc_per_node 1 --master_port 1054 run.py --dataset multiarith
# CUDA_VISIBLE_DECIDES=1 torchrun --nproc_per_node 1 --master_port 1054 run.py --dataset gsm8k
# CUDA_VISIBLE_DECIDES=1 python -m torch.distributed.run run.py --dataset gsm8k
# CUDA_VISIBLE_DECIDES=0 python -m torch.distributed.run run.py --dataset multiarith