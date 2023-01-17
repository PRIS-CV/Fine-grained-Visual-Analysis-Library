from typing import Iterable
from torch.autograd import Variable
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import torch
import torch.functional as F

from . import update_function
from fgvclib.utils.logger import Logger



@update_function("update_swint")
def update_swint(model: nn.Module, optimizer: Optimizer, pbar:Iterable, lr_schedule=None,
    strategy:str="update_swint", use_cuda:bool=True, logger:Logger=None, 
    epoch:int=None, total_epoch:int=None, amp:bool=False, use_selection=False, cfg=None, **kwargs,
):  
    scaler = GradScaler()
    lambda_s = cfg.lambda_s
    lambda_n = cfg.lambda_n
    num_classes = cfg.MODEL.CLASS_NUM
    use_fpn = cfg.MODEL.ARGS["use_fpn"]
    lambda_b = cfg.lambda_b
    use_combiner = cfg.use_combiner
    lambda_c = cfg.lambda_c
    update_freq = cfg.update_freq

    optimizer.zero_grad()
    total_batchs = len(pbar)  # just for log
    show_progress = [x / 10 for x in range(11)]  # just for log
    progress_i = 0
    for batch_id, (datas, labels) in enumerate(pbar):
        model.train()
        """ = = = = adjust learning rate = = = = """
        iterations = epoch * len(pbar) + batch_id

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iterations]

        batch_size = labels.size(0)

        """ = = = = forward and calculate loss = = = = """
        datas, labels = datas.cuda(), labels.cuda()
        datas, labels = Variable(datas), Variable(labels)
        with autocast():
            """
            [Model Return]
                FPN + Selector + Combiner --> return 'layer1', 'layer2', 'layer3', 'layer4', ...(depend on your setting)
                    'preds_0', 'preds_1', 'comb_outs'
                FPN + Selector --> return 'layer1', 'layer2', 'layer3', 'layer4', ...(depend on your setting)
                    'preds_0', 'preds_1'
                FPN --> return 'layer1', 'layer2', 'layer3', 'layer4' (depend on your setting)
                ~ --> return 'ori_out'

            [Retuen Tensor]
                'preds_0': logit has not been selected by Selector.
                'preds_1': logit has been selected by Selector.
                'comb_outs': The prediction of combiner.
            """
            outs = model(datas)

            loss = 0.
            for name in outs:

                if "select_" in name:
                    if not use_selection:
                        raise ValueError("Selector not use here.")
                    if lambda_s != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, num_classes).contiguous()
                        loss_s = nn.CrossEntropyLoss()(logit,
                                                       labels.unsqueeze(1).repeat(1, S).flatten(0))
                        loss += lambda_s * loss_s
                    else:
                        loss_s = 0.0

                elif "drop_" in name:
                    if not use_selection:
                        raise ValueError("Selector not use here.")

                    if lambda_n != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, num_classes).contiguous()
                        n_preds = nn.Tanh()(logit)
                        labels_0 = torch.zeros([batch_size * S, num_classes]) - 1
                        labels_0 = labels_0.cuda()
                        loss_n = nn.MSELoss()(n_preds, labels_0)
                        loss += lambda_n * loss_n
                    else:
                        loss_n = 0.0

                elif "layer" in name:
                    if not use_fpn:
                        raise ValueError("FPN not use here.")
                    if lambda_b != 0:
                        ### here using 'layer1'~'layer4' is default setting, you can change to your own
                        loss_b = nn.CrossEntropyLoss()(outs[name].mean(1), labels)
                        loss += lambda_b * loss_b
                    else:
                        loss_b = 0.0

                elif "comb_outs" in name:
                    if not use_combiner:
                        raise ValueError("Combiner not use here.")

                    if lambda_c != 0:
                        loss_c = nn.CrossEntropyLoss()(outs[name], labels)
                        loss += lambda_c * loss_c

                elif "ori_out" in name:
                    loss_ori = F.cross_entropy(outs[name], labels)
                    loss += loss_ori

            loss /= update_freq

        """ = = = = calculate gradient = = = = """
        if amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        """ = = = = update model = = = = """
        if (batch_id + 1) % update_freq == 0:
            if amp:
                scaler.step(optimizer)
                scaler.update()  # next batch
            else:
                optimizer.step()
            optimizer.zero_grad()
       