import logging
import os
import sys
import os.path as op

def setup_logger(name, save_dir, if_train, distributed_rank=0):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

    # luôn tạo folder
    if not op.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # rank0: log console + file
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # mọi rank: log file riêng để debug
    if if_train:
        log_name = f"train_log_rank{distributed_rank}.txt"
        fh = logging.FileHandler(os.path.join(save_dir, log_name), mode='w')
    else:
        log_name = f"test_log_rank{distributed_rank}.txt"
        fh = logging.FileHandler(os.path.join(save_dir, log_name), mode='a')

    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger