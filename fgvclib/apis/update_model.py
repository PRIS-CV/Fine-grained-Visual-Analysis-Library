from ..utils.update_strategy import get_update_strategy

def update_model(model, optimizer, pbar, strategy="general_updating", use_cuda=True, logger=None):
    model.train()
    mean_loss = 0.
    for batch_idx, train_data in enumerate(pbar):
        losses_info = get_update_strategy(strategy)(model, train_data, optimizer, use_cuda, logger)
        mean_loss = (mean_loss * batch_idx + losses_info['iter_loss']) / (batch_idx + 1)
        losses_info.update({"mean_loss": mean_loss})
        pbar.set_postfix(losses_info)