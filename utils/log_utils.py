import torch
from Metis import BitLinear

@torch.no_grad()
def log_sr(args, model, writer, batch):
    for name, p in model.decoders.named_parameters():
        if ("weight" in name and not "ln" in name):
            _, s, _ = torch.svd_lowrank(p, q=1, niter=4)
            writer.add_scalar(f"sr/{name}", torch.sum(p ** 2) / (s[0] ** 2), batch)