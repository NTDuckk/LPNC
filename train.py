import os
import os.path as op
import time
import random
import warnings

import torch
import numpy as np

from datasets import build_dataloader
from processor.processor import do_train, do_inference  # do_inference chưa dùng ở file này nhưng cứ giữ
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize  # có thể không dùng get_rank nữa, nhưng giữ để không phá nơi khác

warnings.filterwarnings("ignore")


def set_seed(seed=1):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    args = get_args()

    # ====== lấy rank / world_size / local_rank từ torchrun env (nếu có) ======
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(getattr(args, "local_rank", 0))))

    # expose for other modules
    args.rank = rank
    args.world_size = world_size
    args.local_rank = local_rank

    # distributed chỉ bật khi WORLD_SIZE > 1
    args.distributed = world_size > 1

    # để không bị NameError ở logger "Using {} GPUs"
    # (thực chất đây là số process/DDP workers)
    num_gpus = world_size

    # seed theo rank
    set_seed(1 + rank)

    name = args.name

    # ====== chọn device + init process group (nếu DDP) ======
    if torch.cuda.is_available():
        if args.distributed:
            ndev = torch.cuda.device_count()
            if local_rank >= ndev:
                raise RuntimeError(
                    f"local_rank={local_rank} nhưng máy chỉ có {ndev} GPU. "
                    f"Bạn đang torchrun nhiều process hơn số GPU."
                )

            torch.cuda.set_device(local_rank)

            # thêm device_id để tránh warning "Guessing device ID..."
            try:
                torch.distributed.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    device_id=torch.device("cuda", local_rank),
                )
            except TypeError:
                # PyTorch cũ không hỗ trợ device_id
                torch.distributed.init_process_group(backend="nccl", init_method="env://")

            synchronize()
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")

    # ====== output dir + logger ======
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if args.distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        obj = [cur_time] if rank == 0 else [None]
        torch.distributed.broadcast_object_list(obj, src=0)
        cur_time = obj[0]
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}_{args.loss_names}')

    logger = setup_logger(
        'LPNC',
        save_dir=args.output_dir,
        if_train=args.training,
        distributed_rank=rank
    )
    logger.info(f"Using {num_gpus} processes (WORLD_SIZE={world_size})")
    logger.info(f"rank={rank}, local_rank={local_rank}, cuda_device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    logger.info(str(args).replace(',', '\n'))

    save_train_configs(args.output_dir, args)
    os.makedirs(op.join(args.output_dir, "img"), exist_ok=True)

    # ====== build dataloader + model ======
    train_loader, val_img_loader, val_txt_loader, refer_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)

    logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0))
    model.to(device)

    # ====== finetune load (nếu có) ======
    if args.finetune:
        logger.info("loading {} model".format(args.finetune))
        param_dict = torch.load(args.finetune, map_location='cpu')['model']
        for k in list(param_dict.keys()):
            refine_k = k.replace('module.', '')
            param_dict[refine_k] = param_dict[k].detach().clone()
            del param_dict[k]
        model.load_state_dict(param_dict, False)

    # ====== wrap DDP ======
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,   # ✅ FIX lỗi "Expected to have finished reduction..."
        )

    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    # ====== checkpointer / evaluator ======
    is_master = (rank == 0)
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader, refer_txt_loader, args)

    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch']
        logger.info(f"===================>start {start_epoch}")

    # ====== train ======
    try:
        do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)
    finally:
        # tránh warning "destroy_process_group() was not called"
        if args.distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()