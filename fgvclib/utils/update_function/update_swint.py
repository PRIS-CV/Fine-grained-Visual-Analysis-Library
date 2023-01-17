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
    epoch:int=None, total_epoch:int=None, amp:bool=False, use_selection=False, **kwargs,
):
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
                    if args.lambda_s != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, args.num_classes).contiguous()
                        loss_s = nn.CrossEntropyLoss()(logit,
                                                       labels.unsqueeze(1).repeat(1, S).flatten(0))
                        loss += args.lambda_s * loss_s
                    else:
                        loss_s = 0.0

                elif "drop_" in name:
                    if not use_selection:
                        raise ValueError("Selector not use here.")

                    if args.lambda_n != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, args.num_classes).contiguous()
                        n_preds = nn.Tanh()(logit)
                        labels_0 = torch.zeros([batch_size * S, args.num_classes]) - 1
                        labels_0 = labels_0.cuda()
                        loss_n = nn.MSELoss()(n_preds, labels_0)
                        loss += args.lambda_n * loss_n
                    else:
                        loss_n = 0.0

                elif "layer" in name:
                    if not args.use_fpn:
                        raise ValueError("FPN not use here.")
                    if args.lambda_b != 0:
                        ### here using 'layer1'~'layer4' is default setting, you can change to your own
                        loss_b = nn.CrossEntropyLoss()(outs[name].mean(1), labels)
                        loss += args.lambda_b * loss_b
                    else:
                        loss_b = 0.0

                elif "comb_outs" in name:
                    if not args.use_combiner:
                        raise ValueError("Combiner not use here.")

                    if args.lambda_c != 0:
                        loss_c = nn.CrossEntropyLoss()(outs[name], labels)
                        loss += args.lambda_c * loss_c

                elif "ori_out" in name:
                    loss_ori = F.cross_entropy(outs[name], labels)
                    loss += loss_ori

            loss /= args.update_freq

        """ = = = = calculate gradient = = = = """
        if amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        """ = = = = update model = = = = """
        if (batch_id + 1) % args.update_freq == 0:
            if args.use_amp:
                scaler.step(optimizer)
                scaler.update()  # next batch
            else:
                optimizer.step()
            optimizer.zero_grad()

        """ log """
        if (batch_id + 1) % args.log_freq == 0:
            model.eval()
            msg = {}
            msg['info/epoch'] = epoch + 1
            msg['info/lr'] = get_lr(optimizer)
            msg['loss'] = loss.item()
            cal_train_metrics(args, msg, outs, labels, batch_size)
            logger(msg)

        train_progress = (batch_id + 1) / total_batchs
        # print(train_progress, show_progress[progress_i])
        if train_progress > show_progress[progress_i]:
            print(".." + str(int(show_progress[progress_i] * 100)) + "%", end='', flush=True)
            progress_i += 1






@torch.no_grad()
def cal_train_metrics(msg: dict, outs: dict, labels: torch.Tensor, batch_size: int, use_fpn, use_selection, use_combiner):
    """
    only present top-1 training accuracy
    """

    total_loss = 0.0

    if use_fpn:
        for i in range(1, 5):
            acc = top_k_corrects(outs["layer"+str(i)].mean(1), labels, tops=[1])["top-1"] / batch_size
            acc = round(acc * 100, 2)
            msg["train_acc/layer{}_acc".format(i)] = acc
            loss = F.cross_entropy(outs["layer"+str(i)].mean(1), labels)
            msg["train_loss/layer{}_loss".format(i)] = loss.item()
            total_loss += loss.item()

    if use_selection:
        for name in outs:
            if "select_" not in name:
                continue
            B, S, _ = outs[name].size()
            logit = outs[name].view(-1, args.num_classes)
            labels_0 = labels.unsqueeze(1).repeat(1, S).flatten(0)
            acc = top_k_corrects(logit, labels_0, tops=[1])["top-1"] / (B*S)
            acc = round(acc * 100, 2)
            msg["train_acc/{}_acc".format(name)] = acc
            labels_0 = torch.zeros([B * S, num_classes]) - 1
            labels_0 = labels_0.cuda()
            loss = F.mse_loss(F.tanh(logit), labels_0)
            msg["train_loss/{}_loss".format(name)] = loss.item()
            total_loss += loss.item()

        for name in outs:
            if "drop_" not in name:
                continue
            B, S, _ = outs[name].size()
            logit = outs[name].view(-1, num_classes)
            labels_1 = labels.unsqueeze(1).repeat(1, S).flatten(0)
            acc = top_k_corrects(logit, labels_1, tops=[1])["top-1"] / (B*S)
            acc = round(acc * 100, 2)
            msg["train_acc/{}_acc".format(name)] = acc
            loss = F.cross_entropy(logit, labels_1)
            msg["train_loss/{}_loss".format(name)] = loss.item()
            total_loss += loss.item()

    if use_combiner:
        acc = top_k_corrects(outs['comb_outs'], labels, tops=[1])["top-1"] / batch_size
        acc = round(acc * 100, 2)
        msg["train_acc/combiner_acc"] = acc
        loss = F.cross_entropy(outs['comb_outs'], labels)
        msg["train_loss/combiner_loss"] = loss.item()
        total_loss += loss.item()

    if "ori_out" in outs:
        acc = top_k_corrects(outs["ori_out"], labels, tops=[1])["top-1"] / batch_size
        acc = round(acc * 100, 2)
        msg["train_acc/ori_acc"] = acc
        loss = F.cross_entropy(outs["ori_out"], labels)
        msg["train_loss/ori_loss"] = loss.item()
        total_loss += loss.item()

    msg["train_loss/total_loss"] = total_loss


@torch.no_grad()
def top_k_corrects(preds: torch.Tensor, labels: torch.Tensor, tops: list = [1, 3, 5]):
    """
    preds: [B, C] (C is num_classes)
    labels: [B, ]
    """
    if preds.device != torch.device('cpu'):
        preds = preds.cpu()
    if labels.device != torch.device('cpu'):
        labels = labels.cpu()
    tmp_cor = 0
    corrects = {"top-"+str(x):0 for x in tops}
    sorted_preds = torch.sort(preds, dim=-1, descending=True)[1]
    for i in range(tops[-1]):
        tmp_cor += sorted_preds[:, i].eq(labels).sum().item()
        # records
        if "top-"+str(i+1) in corrects:
            corrects["top-"+str(i+1)] = tmp_cor
    return corrects

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        if param_group["lr"] is not None:
            return param_group["lr"]


