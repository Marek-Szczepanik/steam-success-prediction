"""Dočasný test rychlosti GPU: varianty trénovacího kroku modelu A (po použití smazat)."""
import time

import torch

from models.hybrid import HybridModel
from models.model_a import CNNEncoder


def measure(name: str, amp: bool, steps: int = 100):
    """Změří čas `steps` trénovacích kroků na náhodných datech o velikosti jedné dávky."""
    encoder = CNNEncoder()
    model = HybridModel(encoder, encoder.out_dim, 77).cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    meta = torch.randn(64, 77, device="cuda")
    images = torch.rand(64, 4, 3, 128, 128, device="cuda")
    y = torch.randn(64, device="cuda")

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pred = model(meta, images)
        loss = torch.nn.functional.huber_loss(pred.float(), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # rozehřátí: kompilace a výběr kernelů se do měření nepočítá
    for _ in range(10):
        step()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    print(f"{name}: {time.time() - start:.1f} s / {steps} kroků")


if __name__ == "__main__":
    measure("původní (float32)", amp=False)
    torch.backends.cudnn.benchmark = True
    measure("benchmark", amp=False)
    measure("benchmark + bfloat16", amp=True)
