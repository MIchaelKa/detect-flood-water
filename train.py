import torch
import time

# TODO: only for np.array(flood_id_batch)
# does it make sense to separate files which has dependencies to torch, numpy, pandas etc.
import numpy as np

from metrics import AverageMeter, IoUMeter

from utils import format_time

def set_encoder_grad(model, requires_grad):
    for param in model.encoder.parameters():
        param.requires_grad = requires_grad

def compute_prediction(output):
    preds = torch.softmax(output, dim=1)[:, 1]
    # preds *= mask
    preds = (preds > 0.5) * 1
    return preds

def get_next_valid_iter(valid_iters):
    if len(valid_iters) > 0:
        return valid_iters.pop(0)
    else:
        return -1

def validate(model, device, valid_loader, criterion, verbose=True, print_every=10):

    t0 = time.time()

    model.eval()
    
    loss_meter = AverageMeter()
    score_meter = IoUMeter()
    outputs = []

    with torch.no_grad():
        for iter_num, data_dict in enumerate(valid_loader):

            x_batch = data_dict['chip']
            y_batch = data_dict['label']
            # mask_batch = data_dict['mask']
            flood_id_batch = data_dict['flood_id']

            x_batch = x_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device, dtype=torch.long)
            # mask_batch = mask_batch.to(device)
            flood_id_batch = np.array(flood_id_batch)
            
            output = model(x_batch)
            loss = criterion(output, y_batch)
            
            # Update loss meter
            loss_item = loss.item()
            loss_meter.update(loss_item)

            # Update score meter
            preds = compute_prediction(output)
            score_meter.update_with_flood_id(preds, y_batch, flood_id_batch)

            # Save outputs
            probs = torch.softmax(output, dim=1)[:, 1]
            outputs.append(probs.reshape(-1).detach().cpu().numpy())

            if verbose and iter_num % print_every == 0:
                loss_avg = loss_meter.compute_average()
                v_score = score_meter.compute_score()

                print('[valid] iter: {:>4d}, loss = {:.5f}, score = {:.5f}, time: {}'
                    .format(iter_num, loss_avg, v_score, format_time(time.time() - t0)))
   
    return loss_meter, score_meter, outputs

def train_model(
    model,
    device,
    data_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    max_iter,
    unfreeze_iter,
    valid_iters=[],
    save_model=False,
    model_save_name='model',
    verbose=True,
    print_every=10
    ):

    t0 = time.time()

    print('[train] started...')

    train_loss_meter = AverageMeter()
    train_score_meter = IoUMeter()

    train_loss_history = []
    train_score_history = []

    valid_loss_history = []
    valid_score_history = []
    valid_score_by_flood_id = {}

    lr_history = []

    valid_best_score = 0
    best_score_iter = 0

    model.train()

    if unfreeze_iter > 0:
        print('[train] freeze encoder.')
        set_encoder_grad(model, False)
    
    data_loader_iter = iter(data_loader)

    valid_iters_copy = valid_iters.copy()
    valid_iter_num = get_next_valid_iter(valid_iters_copy)

    for iter_num in range(0, max_iter):

        try:
            data_dict = next(data_loader_iter)
        except StopIteration:
            data_loader_iter = iter(data_loader)
            data_dict = next(data_loader_iter)

        # TODO: Check if using a dictionary do not contribute to training time much
        # id_batch = data_dict['chip_id']
        x_batch = data_dict['chip']
        y_batch = data_dict['label']
        # mask_batch = data_dict['mask']

        x_batch = x_batch.to(device, dtype=torch.float32)
        y_batch = y_batch.to(device, dtype=torch.long)
        # mask_batch = mask_batch.to(device)

        # print(x_batch.shape)

        output = model(x_batch)

        loss = criterion(output, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update loss meter
        loss_item = loss.item()
        train_loss_meter.update(loss_item)

        # Update score meter
        # Does it contribute to training time much?
        preds = compute_prediction(output)
        train_score_meter.update(preds, y_batch)

        # Get the last learning rate computed by the scheduler
        lr_history.append(scheduler.get_last_lr())
        # Scheduler update
        scheduler.step()           

        # if verbose and iter_num % print_every == 0:
        #     t_loss_avg = train_loss_meter.compute_average()
        #     t_score = train_score_meter.compute_score()

        #     print('[train] iter: {:>4d}, loss = {:.5f}, score = {:.5f}, time: {}'
        #         .format(iter_num, t_loss_avg, t_score, format_time(time.time() - t0)))

        if iter_num == unfreeze_iter and unfreeze_iter > 0:
            print('[train] unfreeze encoder.')
            set_encoder_grad(model, True)

        if iter_num == valid_iter_num:
            valid_iter_num = get_next_valid_iter(valid_iters_copy)

            v_loss_meter, v_score_meter, outputs = validate(model, device, valid_loader, criterion, verbose=False, print_every=5)

            # TODO: one more train meters to reset it here?

            t_loss_avg = train_loss_meter.compute_average()
            t_score = train_score_meter.compute_score()
            
            train_loss_history.append(t_loss_avg)
            train_score_history.append(t_score)

            v_loss_avg = v_loss_meter.compute_average()
            v_score = v_score_meter.compute_score()
            
            valid_loss_history.append(v_loss_avg)
            valid_score_history.append(v_score)

            v_score_by_flood_id = v_score_meter.compute_score_by_flood_id()
            for i, (k, v) in enumerate(v_score_by_flood_id.items()):
                if k not in valid_score_by_flood_id:
                    valid_score_by_flood_id[k] = []
                valid_score_by_flood_id[k].append(v)

            # v_loss_avg is better?
            if v_score > valid_best_score:
                valid_best_score = v_score
                best_score_iter = iter_num

                if save_model:
                    # torch.save(model.state_dict(), f'pth/{model_save_name}_{iter_num}.pth')
                    torch.save(model.state_dict(), f'pth/{model_save_name}.pth')
                    
            # TODO: move out and see performance and time
            model.train()

            if verbose:
                print('[train] iter: {:>4d}, loss = {:.5f}, score = {:.5f}, time: {}'
                    .format(iter_num, t_loss_avg, t_score, format_time(time.time() - t0)))
                print('[valid] iter: {:>4d}, loss = {:.5f}, score = {:.5f}, time: {}'
                    .format(iter_num, v_loss_avg, v_score, format_time(time.time() - t0)))
                print('[valid] iter: {:>4d}, score = {}'.format(iter_num, v_score_by_flood_id))
                print('')

    if verbose:
        print('[valid] best score = {:.5f}, iter: {:>4d}'.format(valid_best_score, best_score_iter))
        print('[train] finished for: {}'.format(format_time(time.time() - t0)))

    train_info = {
        'train_loss_meter' : train_loss_meter,
        'train_score_meter' : train_score_meter,

        'valid_loss_meter' : v_loss_meter,
        'valid_score_meter' : v_score_meter,
        'valid_outputs' : outputs,

        'train_loss_history' : train_loss_history,
        'train_score_history' : train_score_history,

        'valid_loss_history' : valid_loss_history,
        'valid_score_history' : valid_score_history,
        'valid_score_by_flood_id' : valid_score_by_flood_id,

        'lr_history' : lr_history,

        'valid_iters' : valid_iters,

        'best_score' : valid_best_score,
        'best_score_iter' : best_score_iter,
    }

    return train_info