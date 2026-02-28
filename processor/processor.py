import logging
import os
import time
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize


def _dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _world_size() -> int:
    return dist.get_world_size() if _dist_avail_and_initialized() else 1


def _get_device(args) -> torch.device:
    if torch.cuda.is_available():
        # With torchrun, each process should have its own CUDA device set via torch.cuda.set_device(local_rank).
        # Still, we explicitly pin to local_rank to avoid surprises.
        local_rank = int(getattr(args, "local_rank", 0))
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _move_to_device(x: Any, device: torch.device):
    """Move only tensors to device; keep lists/strings/None as-is (matches permissive collate)."""
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    return x


@torch.no_grad()
def _reduce_tensor(t: torch.Tensor, average: bool = True) -> torch.Tensor:
    if not _dist_avail_and_initialized():
        return t
    rt = t.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if average:
        rt /= float(_world_size())
    return rt


@torch.no_grad()
def _reduce_number(x: Any, device: torch.device) -> float:
    """Reduce a scalar-like value (tensor/float/int) to a float averaged across ranks."""
    if isinstance(x, torch.Tensor):
        t = x.detach()
        if t.numel() != 1:
            t = t.mean()
    else:
        t = torch.tensor(float(x), device=device)
    t = t.float()
    t = _reduce_tensor(t, average=True)
    return float(t.item())


def _set_epoch_for_sampler(train_loader, epoch: int) -> None:
    """Call set_epoch(epoch) for DistributedSampler or any sampler that supports it."""
    # 1) DataLoader(sampler=...)
    sampler = getattr(train_loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

    # 2) DataLoader(batch_sampler=BatchSampler(...))
    batch_sampler = getattr(train_loader, "batch_sampler", None)
    if batch_sampler is not None:
        inner = getattr(batch_sampler, "sampler", None)
        if inner is not None and hasattr(inner, "set_epoch"):
            inner.set_epoch(epoch)


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    logger = logging.getLogger("LPNC.train")
    log_period = args.log_period
    eval_period = args.eval_period
    device = _get_device(args)
    num_epoch = args.num_epoch

    arguments: Dict[str, Any] = {"num_epoch": num_epoch, "iteration": 0}

    meters: Dict[str, AverageMeter] = {
        "loss": AverageMeter(),
        "supid_loss": AverageMeter(),
        "cotrl_loss": AverageMeter(),
        "cid_loss": AverageMeter(),
    }

    tb_writer: Optional[SummaryWriter] = None
    if get_rank() == 0:
        tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()

        if getattr(args, "distributed", False):
            _set_epoch_for_sampler(train_loader, epoch)

        model.train()
        # some models use model.epoch internally
        setattr(model, "epoch", epoch)
        if hasattr(model, "module"):
            setattr(model.module, "epoch", epoch)

        for n_iter, batch in enumerate(train_loader):
            # Move to device (only tensors)
            batch = {k: _move_to_device(v, device) for k, v in batch.items()}

            ret = model(batch)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            # ---- logging metrics (reduced across ranks) ----
            # NOTE: meters are for logging only; we do not alter training loss/backward
            local_bs = None
            for _v in batch.values():
                if torch.is_tensor(_v) and _v.dim() > 0:
                    local_bs = int(_v.size(0))
                    break
            local_bs = local_bs or 1
            global_bs = local_bs * _world_size()

            loss_avg = _reduce_number(total_loss, device)
            meters["loss"].update(loss_avg, global_bs)

            meters["supid_loss"].update(_reduce_number(ret.get("supid_loss", 0.0), device), global_bs)
            meters["cotrl_loss"].update(_reduce_number(ret.get("cotrl_loss", 0.0), device), global_bs)
            meters["cid_loss"].update(_reduce_number(ret.get("cid_loss", 0.0), device), global_bs)

            if get_rank() == 0 and (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                # scheduler.get_lr() exists in your codebase usage
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

        # ---- epoch end ----
        if get_rank() == 0 and tb_writer is not None:
            tb_writer.add_scalar("lr", scheduler.get_lr()[0], epoch)
            if "temperature" in ret:
                try:
                    tb_writer.add_scalar("temperature", float(ret["temperature"]), epoch)
                except Exception:
                    pass
            for k, v in meters.items():
                if v.avg > 0:
                    tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()

        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / max(1, (n_iter + 1))
            # train_loader.batch_size may not exist when using batch_sampler
            bs_for_speed = getattr(train_loader, "batch_size", None) or global_bs
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch, time_per_batch, bs_for_speed / time_per_batch
                )
            )

        # ---- evaluation (keep all ranks in sync to avoid DDP stalls) ----
        if epoch % eval_period == 0:
            if getattr(args, "distributed", False):
                synchronize()

            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if getattr(args, "distributed", False):
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()

                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)

            if getattr(args, "distributed", False):
                synchronize()

    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments.get('epoch', -1)}")
        if tb_writer is not None:
            tb_writer.close()


def do_inference(model, test_img_loader, test_txt_loader, refer_loader, args):
    logger = logging.getLogger("LPNC.test")
    if get_rank() == 0:
        logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader, refer_loader, args)
    m = model.module if hasattr(model, 'module') else model
    top1 = evaluator.eval(m.eval())
    return top1