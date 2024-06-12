# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import pickle
import utils

def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # print(data_iter_step)
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):

                if epoch < param_group['fix_step']:
                    param_group["lr"] = 0.
                elif lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
                # print(epoch, param_group['fix_step'], param_group["lr"])
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                # print(samples.size())
                output = model(samples)
                # print(len(output))
                loss, loss_part = criterion(samples, output, targets)
                softmax_output = output[0].softmax(dim=-1)
                class_acc = (output[0].max(-1)[-1] == targets).float().mean()

                entropy = -1 * (softmax_output * torch.log(softmax_output)).mean(dim=-1)

                pred_entropy = -1 * class_acc * entropy

                pred_index = torch.nonzero(pred_entropy).squeeze()
                # print('pred_entropy')
                # not_pred_index = torch.nonzero(not_pred_entropy).squeeze()

                pred_entropy = -1 * pred_entropy[pred_index]
                entropy_loss = 1000*pred_entropy.mean()
                loss += entropy_loss

                # print(f'loss {loss} cls {loss_part[0]+ loss_part[1]+loss_part[2]+entropy_loss} {entropy_loss}')

                # print(len(loss)) #.size())
        else: # full precision
            output = model(samples)
            loss, loss_part = criterion(samples, output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)

            # print(f'######## Shared layers ########')
            # # print(f'mlp1 {model.blocks[0].mlp.fc1.weight[0,0].requires_grad}, mlp2 {model.blocks[1].mlp.fc1.weight[0,0].requires_grad}, mlp3 {model.blocks[2].mlp.fc1.weight[0,0].requires_grad}')
            # print(f"norm1 {model.blocks[0].norm1.weight.sum()}, norm2 {model.blocks[1].norm1.weight.sum()}, norm3 {model.blocks[2].norm1.weight.data[0]},"
            #       # f" qkv1 {model.blocks[0].attn.qkv.weight.data[0,0]}, qkv1 {model.blocks[1].attn.qkv.weight.data[0,0]}, qkv1 {model.blocks[2].attn.qkv.weight.data[0,0]}"
            #       # f"proj_attn1 {model.blocks[0].attn.proj.weight.data[0,0]}, proj_attn2 {model.blocks[1].attn.proj.weight.data[0,0]}, proj_attn3 {model.blocks[2].attn.proj.weight.data[0,0]} "
            #       )
            # print(f'######## Shared End ########')
            #
            #
            # print(f"in_conv {model.score_predictor[0].in_conv[1].weight.data[0,0]}")
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            # print(len(output), output[0].size())
            class_acc = (output[0].max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(cls_loss=loss_part[0], head="loss")
            log_writer.update(ratio_loss=loss_part[1], head="loss")
            log_writer.update(cls_distill_loss=loss_part[2], head="loss")
            log_writer.update(token_distill_loss=loss_part[3], head="loss")
            log_writer.update(layer_mse_loss=loss_part[4], head="loss")
            log_writer.update(feat_distill_loss=loss_part[5], head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    i = 0
    empirical_score = 0
    entropy_pred_list = []
    entropy_not_pred_list = []
    softmax_list, class_acc_list = [],[]
    for batch in metric_logger.log_every(data_loader, 10, header):

        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)


                softmax_output = output.softmax(dim=-1).cpu()
                entropy = -1* (softmax_output * torch.log(softmax_output)).mean(dim=-1)
                # print(entropy.mean())
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))


        class_acc = (output.max(-1)[-1] == target).float().cpu()

        pred = class_acc
        not_pred = 1 - pred

        pred_entropy = -1 * pred * entropy
        not_pred_entropy = -1 * not_pred * entropy

        pred_index = torch.nonzero(pred_entropy).squeeze()
        not_pred_index = torch.nonzero(not_pred_entropy).squeeze()

        pred_entropy = -1*pred_entropy[pred_index]
        not_pred_entropy = -1*not_pred_entropy[not_pred_index]


        print(f'mean_entropy_classified {pred_entropy.mean()} not classified {not_pred_entropy.mean()}')
        #
        # if i < 10:
        # softmax_list.append(softmax_output.cpu())
        # class_acc_list.append(class_acc.cpu())



        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        i += 1

    # with open('softmax_list'+str(epoch)+'.pkl', 'wb') as file:
    #     pickle.dump(softmax_list, file)
    # with open('class_acc_list'+str(epoch)+'.pkl', 'wb') as file:
    #     pickle.dump(class_acc_list, file)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
