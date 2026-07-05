"""데이터 병렬화의 핵심인 All-Reduce(그래디언트 동기화)를 CPU 다중 프로세스로 시뮬레이션"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(0)  # 모든 프로세스가 동일한 초기 가중치를 갖도록
    model = nn.Linear(4, 2)

    torch.manual_seed(rank)  # 프로세스(=GPU)마다 다른 데이터를 갖도록
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)

    output = model(x)
    loss = ((output - y) ** 2).mean()
    loss.backward()

    grad_before = model.weight.grad.clone()

    # All-Reduce: 모든 프로세스의 그래디언트를 더한 뒤 프로세스 수로 나눔 (평균)
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= world_size

    if rank == 0:
        print(f"[프로세스 {rank}] 동기화 전 그래디언트 일부: {grad_before.flatten()[:3]}")
        print(f"[프로세스 {rank}] 동기화 후(전체 평균) 그래디언트 일부: {model.weight.grad.flatten()[:3]}")

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 4
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
