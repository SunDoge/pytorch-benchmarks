"""
测试计算metric的位置对性能的影响

没有明显影响
"""

from tqdm import tqdm, trange
import torch
from torchvision.models import resnet18
from torch import nn, Tensor
from utils.metrics import accuracy
import time


def bench_normal(model: nn.Module, x: Tensor, n=100):

    for _ in trange(n):

        # Transfer data to gpu
        x_cuda = x.cuda(non_blocking=True)
        target = torch.ones(x_cuda.size(0), device=x_cuda.device)

        # Computation
        y = model(x_cuda)
        y.sum().backward()
        model.zero_grad()

        output = (y, target)

        # Compute metrics
        acc1, acc5 = accuracy(*output, topk=(1, 5))
        # time.sleep(0.1)

        torch.cuda.synchronize()


def bench_history(model: nn.Module, x: Tensor, n=100):

    # Precompute for fair comparision
    x1 = x.cuda()
    y1 = model(x1)
    target1 = torch.ones(x1.size(0), device=x1.device)
    output = (y1, target1)

    for _ in trange(n):
        # Transfer data to gpu
        x_cuda = x.cuda(non_blocking=True)
        target = torch.ones(x_cuda.size(0), device=x_cuda.device)

        # Compute metrics on history
        if output is not None:
            acc1, acc5 = accuracy(*output, topk=(1, 5))
            # time.sleep(0.1)

        # Compute and update history
        y = model(x_cuda)
        y.sum().backward()
        model.zero_grad()
        output = (y, target)

        torch.cuda.synchronize()


def main():
    model = resnet18()
    x = torch.rand(2, 3, 224, 224)
    model.cuda()

    print('start warmup')
    bench_normal(model, x, n=10)
    bench_history(model, x, n=10)

    print('=' * 50)

    bench_normal(model, x, n=100)
    bench_history(model, x, n=100)


if __name__ == '__main__':
    main()
