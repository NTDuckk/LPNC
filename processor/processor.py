import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    accum_steps = max(1, args.gradient_accumulation_steps)

    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0
    arguments["epoch"] = start_epoch

    logger = logging.getLogger("LPNC.train")
    logger.info("start training")

    meters = {
        "loss": AverageMeter(),
        "supid_loss": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()

        model.train()
        model.epoch = epoch

        optimizer.zero_grad()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            ret = model(batch)

            total_loss = sum([v for k, v in ret.items() if "loss" in k])
            batch_size = batch["images"].shape[0]

            meters["loss"].update(total_loss.item(), batch_size)

            supid_val = ret.get("supid_loss", 0)
            if torch.is_tensor(supid_val):
                supid_val = supid_val.item()
            meters["supid_loss"].update(supid_val, batch_size)

            # gradient accumulation
            loss_for_backward = total_loss / accum_steps
            loss_for_backward.backward()

            should_step = ((n_iter + 1) % accum_steps == 0) or ((n_iter + 1) == len(train_loader))
            if should_step:
                optimizer.step()
                optimizer.zero_grad()
                synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

        tb_writer.add_scalar("lr", scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar("temperature", ret["temperature"], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()

        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch,
                    time_per_batch,
                    train_loader.batch_size / time_per_batch,
                )
            )

        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)

    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader, refer_loader, args):
    logger = logging.getLogger("LPNC.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader, refer_loader, args)
    top1 = evaluator.eval(model.eval())