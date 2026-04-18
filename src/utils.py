import os
import torch
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)

def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0

def get_rank():
    """Get current process rank."""
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size():
    """Get total number of processes."""
    return dist.get_world_size() if dist.is_initialized() else 1

def unwrap_model(model):
    """Unwrap DDP model to get the underlying model."""
    return model.module if hasattr(model, 'module') else model

def reduce_tensor(tensor, world_size=None):
    """Reduce tensor across all processes."""
    if not dist.is_initialized():
        return tensor
    
    if world_size is None:
        world_size = get_world_size()
    
    # Clone to avoid in-place modification
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / world_size
    return rt

def gather_tensor(tensor):
    """Gather tensor from all processes."""
    if not dist.is_initialized():
        return [tensor]
    
    world_size = get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor.contiguous())
    return tensor_list

def log_on_main(message, logger_obj=None):
    """Log message only on main process."""
    if is_main_process():
        if logger_obj:
            logger_obj.info(message)
        else:
            logger.info(message)