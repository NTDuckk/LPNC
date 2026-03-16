import os
import os.path as op
import torch
import numpy as np
import random
import time
from datasets import build_dataloader
from processor.processor import do_train, do_inference
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize
import warnings
warnings.filterwarnings("ignore")

def print_model_info(model, args, logger):
    try:
        logger.info(f"Pretrained choice: {args.pretrain_choice}")
    except Exception:
        logger.info(f"Pretrained choice: (not set)")
    try:
        logger.info(f"Image size: {args.img_size}")
    except Exception:
        logger.info(f"Image size: (not set)")
    try:
        logger.info(f"Losses / Tasks: {args.loss_names}")
    except Exception:
        logger.info(f"Losses / Tasks: (not set)")

    # Print top-level child modules (concise)
    logger.info("Model top-level modules:")
    for name, module in model.named_children():
        logger.info(f" - {name}: {module.__class__.__name__}")
    # Print a small parameter summary
    try:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {total_params}")
    except Exception:
        pass


def print_optimizer_info(optimizer, model, logger):
    # map parameter id -> name
    name_map = {id(p): n for n, p in model.named_parameters()}

    logger.info("Optimizer parameter groups:")
    for i, g in enumerate(optimizer.param_groups):
        lr = g.get('lr', None)
        wd = g.get('weight_decay', None)
        params = g.get('params', [])
        names = [name_map.get(id(p), '<unknown>') for p in params]
        logger.info(f" - group {i}: lr={lr}, weight_decay={wd}, params_count={len(params)}")
        # list up to first 20 parameter names for readability
        for nm in names[:20]:
            logger.info(f"    {nm}")
        if len(names) > 20:
            logger.info(f"    ... ({len(names)-20} more)")

    # modules that are fully frozen vs trainable
    frozen_modules = []
    trainable_modules = []
    for m_name, m in model.named_children():
        params = list(m.parameters())
        if len(params) == 0:
            continue
        if all(not p.requires_grad for p in params):
            frozen_modules.append(m_name)
        else:
            trainable_modules.append(m_name)

    logger.info(f"Frozen modules: {frozen_modules}")
    logger.info(f"Trainable modules: {trainable_modules}")

    frozen_params = [n for n, p in model.named_parameters() if not p.requires_grad]
    logger.info(f"Frozen parameters count: {len(frozen_params)}")
    for n in frozen_params[:50]:
        logger.info(f"    {n}")
    if len(frozen_params) > 50:
        logger.info(f"    ... ({len(frozen_params)-50} more)")


def print_frozen_modules(model, logger):
    """Print only modules and parameters that are frozen (requires_grad==False)."""
    frozen_modules = []
    for m_name, m in model.named_children():
        params = list(m.parameters())
        if len(params) == 0:
            continue
        if all(not p.requires_grad for p in params):
            frozen_modules.append(m_name)

    logger.info("Frozen modules:")
    if frozen_modules:
        for m in frozen_modules:
            logger.info(f" - {m}")
    else:
        logger.info(" - (none)")

    frozen_params = [n for n, p in model.named_parameters() if not p.requires_grad]
    logger.info(f"Frozen parameters count: {len(frozen_params)}")
    for n in frozen_params[:200]:
        logger.info(f"    {n}")
    if len(frozen_params) > 200:
        logger.info(f"    ... ({len(frozen_params)-200} more)")

def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    args = get_args()
    set_seed(1+get_rank())
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}_{args.loss_names}')
    logger = setup_logger('LPNC', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.output_dir, args)
    if not os.path.isdir(args.output_dir+'/img'):
        os.makedirs(args.output_dir+'/img')
    # get image-text pair datasets dataloader
    # if 'ICFG-PEDES' not in args.dataset_name: #fixed
    #     args.val_dataset = 'val'

        
    train_loader, val_img_loader, val_txt_loader, refer_txt_loader,num_classes = build_dataloader(args)
    model = build_model(args, num_classes)

    # Only print frozen modules/params before moving to device and training
    if 'logger' in globals():
        print_frozen_modules(model, logger)
    else:
        try:
            frozen_params = [n for n, p in model.named_parameters() if not p.requires_grad]
            print(f"Frozen parameters count: {len(frozen_params)}")
        except Exception:
            pass

    logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0))
    model.to(device)
    if args.finetune:
        logger.info("loading {} model".format(args.finetune))
        param_dict = torch.load(args.finetune,map_location='cpu')['model']
        for k in list(param_dict.keys()):
            refine_k = k.replace('module.','')
            param_dict[refine_k] = param_dict[k].detach().clone()
            del param_dict[k]
        model.load_state_dict(param_dict, False)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    # (Only frozen modules were printed earlier; skip verbose optimizer info)


    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader, refer_txt_loader,args)

    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch']
        logger.info(f"===================>start {start_epoch}")

    do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)
