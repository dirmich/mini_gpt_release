"""Simulating All-Reduce (gradient synchronization), the core of data parallelism, using multi-process CPU"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(0)  # Ensure all processes have identical initial weights
    model = nn.Linear(4, 2)

    torch.manual_seed(rank)  # Ensure each process (=GPU) has different data
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)

    output = model(x)
    loss = ((output - y) ** 2).mean()
    loss.backward()

    grad_before = model.weight.grad.clone()

    # All-Reduce: Sum gradients from all processes and divide by the number of processes (average)
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= world_size

    if rank == 0:
        print(f"[Process {rank}] Gradient snippet before synchronization: {grad_before.flatten()[:3]}")
        print(f"[Process {rank}] Gradient snippet after synchronization (global average): {model.weight.grad.flatten()[:3]}")

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 4
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
