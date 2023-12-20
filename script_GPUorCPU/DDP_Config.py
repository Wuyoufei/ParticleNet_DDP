import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def init_ddp(local_rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    world_size = torch.cuda.device_count()
    os.environ['WORLD_SIZE'] = str(world_size)
    #os.environ['LOCAL_RANK'] = str(local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='gloo', rank=local_rank,world_size=world_size,init_method='env://')