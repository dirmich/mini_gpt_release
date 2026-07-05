"""データ並列化の核心であるAll-Reduce（勾配同期）をCPUマルチプロセスでシミュレーション"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(0)  # すべてのプロセスが同一の初期重みを持つように
    model = nn.Linear(4, 2)

    torch.manual_seed(rank)  # プロセス(=GPU)ごとに異なるデータを持つように
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)

    output = model(x)
    loss = ((output - y) ** 2).mean()
    loss.backward()

    grad_before = model.weight.grad.clone()

    # All-Reduce: すべてのプロセスの勾配を合計した後、プロセス数で割る（平均）
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= world_size

    if rank == 0:
        print(f"[Process {rank}] 同期前の勾配の一部: {grad_before.flatten()[:3]}")
        print(f"[Process {rank}] 同期後(全体平均)の勾配の一部: {model.weight.grad.flatten()[:3]}")

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 4
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
