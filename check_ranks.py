import torch, torch.distributed as dist, os

dist.init_process_group('nccl')
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

r  = dist.get_rank()
lr = int(os.environ['LOCAL_RANK'])
d  = torch.cuda.current_device()

print(f"Rank {r} | local_rank {lr} | CUDA device idx {d} | {torch.cuda.get_device_name(d)} | visible_gpus={torch.cuda.device_count()}", flush=True)

dist.destroy_process_group()
