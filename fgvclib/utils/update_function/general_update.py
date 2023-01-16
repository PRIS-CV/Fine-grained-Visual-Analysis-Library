from typing import Iterable
import torch
import torch.nn as nn
from torch.optim import Optimizer


from fgvclib.utils.update_strategy import get_update_strategy
from fgvclib.utils.logger import Logger
from fgvclib.utils.lr_schedules import LRSchedule


def general_update(
    model: nn.Module, optimizer: Optimizer, pbar:Iterable, lr_schedule:LRSchedule=None,
    strategy:str="general_updating", use_cuda:bool=True, logger:Logger=None, 
    epoch:int=None, total_epoch:int=None, amp:bool=False, **kwargs
):
    r"""Update the FGVC model and record losses.

    Args:
        model (nn.Module): The FGVC model.
        optimizer (Optimizer): The Logger object.
        pbar (Iterable): The iterable object provide training data.
        lr_schedule (LRSchedule): The lr schedule updating class.
        strategy (string): The update strategy.
        use_cuda (boolean): Whether to use GPU to train the model.
        logger (Logger): The Logger object.
        epoch (int): The current epoch number.
        total_epoch (int): The total epoch number.
    """

    model.train()
    mean_loss = 0.
    for batch_idx, train_data in enumerate(pbar):
        losses_info = get_update_strategy(strategy)(model, train_data, optimizer, use_cuda, amp)
        mean_loss = (mean_loss * batch_idx + losses_info['iter_loss']) / (batch_idx + 1)
        losses_info.update({"mean_loss": mean_loss})
        logger(losses_info, step=batch_idx)
        pbar.set_postfix(losses_info)
        if lr_schedule.update_level == 'batch_update':
            lr_schedule.step(optimizer=optimizer, batch_idx=batch_idx, batch_size=len(train_data), current_epoch=epoch, total_epoch=total_epoch)
    
    if lr_schedule.update_level == 'epoch_update':
        lr_schedule.step(optimizer=optimizer, current_epoch=epoch, total_epoch=total_epoch)


def update_vitmodel(model: nn.Module, optimizer: Optimizer, scheduler, pbar: Iterable,
                    strategy: str = "vit_updating",
                    use_cuda: bool = True, logger: Logger = None):
    r"""Update the FGVC model and record losses.

    Args:
        model (nn.Module): The FGVC model.
        optimizer (Optimizer): The Logger object.
        scheduler : The scheduler strategy
        pbar (Iterable): A iterable object provide training data.
        strategy (string): The update strategy.
        use_cuda (boolean): Whether to use GPU to train the model.
        logger (Logger): The Logger object.
    """
    model.train()
    mean_loss = 0.
    for batch_idx, train_data in enumerate(pbar):
        losses_info = get_update_strategy(strategy)(model, train_data, optimizer, scheduler, use_cuda)
        mean_loss = (mean_loss * batch_idx + losses_info['iter_loss']) / (batch_idx + 1)
        losses_info.update({"mean_loss": mean_loss})
        logger(losses_info, step=batch_idx)
        pbar.set_postfix(losses_info)


def update_swintModel(args, epoch, model, scaler, amp_context, optimizer, schedule, train_bar, logger: Logger = None):
    optimizer.zero_grad()
    total_batchs = len(train_bar)  # just for log
    show_progress = [x / 10 for x in range(11)]  # just for log
    progress_i = 0
    for batch_id, (datas, labels) in enumerate(train_bar):
        model.train()
        """ = = = = adjust learning rate = = = = """
        iterations = epoch * len(train_bar) + batch_id
        adjust_lr(iterations, optimizer, schedule)

        batch_size = labels.size(0)

        """ = = = = forward and calculate loss = = = = """
        datas, labels = datas.cuda(), labels.cuda()
        datas, labels = Variable(datas), Variable(labels)
        with amp_context():
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
                    if not args.use_selection:
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
                    if not args.use_selection:
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
        if args.use_amp:
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


    