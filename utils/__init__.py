import random
import torch
from torch import Tensor
import os
import numpy as np
from utils.conf import Config
from utils.log import Log

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

def update_max(new: list, old: list) -> list:
    """Update the historical maximum metrics."""
    re = []
    for i,j in zip(new, old):
        if i > j:
            re.append(i)
        else:
            re.append(j)
    return re

def format_epoch(name, epoch, total, results: dict):
    """Format the output string for model's output metrics."""
    result_str = f"Epoch {epoch}/{total}, {name}: "
    for metric in results:
        val = results[metric]
        result_str += f"{metric}={val:.5f}, "
    result_str = result_str[:-2] + '  '
    return result_str

def format_best(epoch, n11, n12, n21, n22, n31, n32):
    """
    Format the output string for model's best metrics (with the historical best).
    n1*: Recall
    n2*: NDCG
    n3*: Precision
    """
    result_str = f"Best epoch: {epoch}, "
    result_str += f"Recall: {n11:.5f}({n12:.5f}), NDCG: {n21:.5f}({n22:.5f}), Precision: {n31:.5f}({n32:.5f}), "
    return result_str

def format_config(config: Config, log: Log):
    """Write configurations to the log."""
    log.info("Configuration Details:")
    for section, options in config.__dict__.items():
        if isinstance(options, dict):
            log.info(f"[{section}]")
            for key, value in options.items():
                log.info(f"  {key}: {value}")
        else:
            log.info(f"{section}: {options}")

def cal_metrics(topk, topk_idxs: np.ndarray, test_u_its: list, users: Tensor) -> tuple[float, float, float]:
    """
    Args:
        topk_idxs (np.ndarray): top-k items' index of test batch users: (test_batch, topk)
        test_u_its (list): test users' interactions: (test_batch, item)
        users (Tensor): test batch users' index: (test_batch, )
    """
    total_recall = total_ndcg = total_precision = 0.0
    for i in range(len(users)):
        u_rec_list = list(topk_idxs[i])
        u_its = test_u_its[users[i]]
        test_num = len(u_its)
        max_dcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(test_num, topk))])
        recall_hits = dcg = precision_hits = 0
        for item in u_its:
            if item in u_rec_list:
                recall_hits += 1
                dcg += np.reciprocal(np.log2(u_rec_list.index(item) + 2))
                precision_hits += 1

        recall = recall_hits / test_num
        ndcg = dcg / max_dcg
        precision = precision_hits / topk

        total_recall += recall
        total_ndcg += ndcg
        total_precision += precision
    return total_recall, total_ndcg, total_precision