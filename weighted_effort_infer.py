# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json

from pathlib import Path
import tqdm
# from timm.data import Mixup
# from timm.models import create_model
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from timm.scheduler import create_scheduler
# from timm.optim import create_optimizer
# from timm.utils import NativeScaler, get_state_dict, ModelEma

# from datasets import build_dataset
# from Raspi_run.engine_raspi import train_one_epoch, evaluate
# from Raspi_run.samplers_raspi import RASampler
# from functools import partial


from model_ip_specific_fast import VisionTransformerDiffPruning
from lvvit_ip_specific_fast import LVViTDiffPruning, LVViT_Teacher

from timm.utils import ModelEma

# from model_extract_attention_only import VisionTransformerDiffPruning
# from model_encoder_prune import VisionTransformerDiffPruning
# from model_dyvit_orig import VisionTransformerDiffPruning

# from models.dylvvit import LVViTDiffPruning
# from models.dyconvnext import AdaConvNeXt
# from models.dyswin import AdaSwinTransformer
import utils

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--model', default='deit-s', type=str, help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', type=utils.str2bool, default=True)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model_path', default='../pretrained/dynamic-vit_384_r0.7.pth', help='resume from checkpoint')
    parser.add_argument('--effort1', default='../pretrained/dynamic-vit_384_r0.7.pth', help='resume from checkpoint')
    parser.add_argument('--effort2', default='../pretrained/dynamic-vit_384_r0.7.pth', help='resume from checkpoint')
    parser.add_argument('--effort3', default='../pretrained/dynamic-vit_384_r0.7.pth', help='resume from checkpoint')

    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--base_rate', type=float, default=0.7)
    parser.add_argument('--keep_ratio', default="1_1_1",
                        help='keep ratio list')
    parser.add_argument('--sharing1', default='X',
                        help='how layers are shared')
    parser.add_argument('--sharing2', default='X',
                        help='how layers are shared')
    parser.add_argument('--sharing3', default='X',
                        help='how layers are shared')

    return parser


import numpy as np


def compute_MAC_cyles(token_rows, token_tile_size, weight_tile_size, token_cols=384, weight_rows=384, weight_cols=384,
                      total_pes=36, n_mults_per_pe=64):
    required_PEs = np.ceil(weight_cols / weight_tile_size) * np.ceil(token_rows / token_tile_size) * np.ceil(
        token_cols / n_mults_per_pe)
    cycles = required_PEs / total_pes

    return cycles


def compute_softmax_cycles(token_rows, n_smax_units=1):
    token_cols = token_rows

    return (token_rows * token_cols) / n_smax_units


def compute_encoder_cycles(n_tokens, token_tile_size, weight_tile_size, n_heads, attn_reuse=False):
    qkv = 3 * compute_MAC_cyles(token_rows=n_tokens, token_tile_size=token_tile_size, weight_tile_size=weight_tile_size)
    qkt = compute_MAC_cyles(token_rows=n_tokens, token_cols=384, weight_rows=384, weight_cols=n_tokens,
                            token_tile_size=token_tile_size, weight_tile_size=weight_tile_size)

    softmax = compute_softmax_cycles(token_rows=n_tokens, n_smax_units=1) * n_heads
    sv = compute_MAC_cyles(token_rows=n_tokens, token_cols=n_tokens, weight_rows=n_tokens, weight_cols=384,
                           token_tile_size=token_tile_size, weight_tile_size=weight_tile_size)
    projection = qkv / 3.
    mlp = qkv / 3. * 4 * 2
    softmax_latency = softmax * 5.5 * 1e-6
    attn_matmul_latency = qkv + qkt + sv + projection

    total_MAC_cycles = (qkv + qkt + sv + projection + mlp) * token_tile_size * weight_tile_size
    if attn_reuse:
        total_latency = mlp * 5.5 * 1e-6
    else:
        total_latency = (total_MAC_cycles + softmax) * 5.5 * 1e-6

    return total_latency, softmax_latency, attn_matmul_latency, mlp

def compute_latency_atr(base_rate=1, reused_encoders=[]):
    latency = 0
    t, t1, t2, t3 = 197, np.ceil(197 * base_rate), np.ceil(197 * base_rate ** 2), np.ceil(197 * base_rate ** 3)
    token_list = [t, t, t, t1, t1, t1, t2, t2, t2, t3, t3, t3]
    _, sm_latency, attn_mm_latency, mlp = compute_encoder_cycles(n_tokens=n_tokens, token_tile_size=66, weight_tile_size=98, n_heads=6, attn_reuse=False)
    softmax_latency, attn_matmul_latency, mlp_latency = 0, 0, 0

    for idx, n_tokens in enumerate(token_list):
        latency += compute_encoder_cycles(n_tokens=n_tokens, token_tile_size=66, weight_tile_size=98, n_heads=6, attn_reuse=idx in reused_encoders)
        if idx not in reused_encoders:
            softmax_latency += sm_latency
            attn_matmul_latency += attn_mm_latency

        mlp_latency += mlp
    return latency, softmax_latency, attn_matmul_latency, mlp_latency


def compute_energy_atr(reused_encoders=[]):

    total_latency, total_softmax_latency, total_attn_matmul_latency, total_mlp_latency = compute_latency_atr(base_rate=1, reused_encoders=reused_encoders)
    softmax_module_power = 2.3
    PE_power = total_pes * 7.3
    PE_SRAM_power = total_pes * 6.7

def main(args):

    cudnn.benchmark = True
    # dataset_val, _ = build_dataset(is_train=False, args=args)

    import load_imagenet100_ffcv as img_ffcv

    batch_size = args.batch_size
    distributed = 0
    in_memory = 1
    num_workers = 4

    val_dataset = '/gpfs/gibbs/project/panda/shared/imagenet_ffcv/val.beton'
    data_loader_val = img_ffcv.create_val_loader(val_dataset, num_workers, batch_size, distributed)
    # data_loader_val = torch.utils.data.DataLoader(
    #     dataset_val,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=False
    # )


    # base_rate = args.base_rate
    # KEEP_RATE1 = [base_rate, base_rate ** 2, base_rate ** 3]
    # KEEP_RATE2 = [base_rate, base_rate - 0.2, base_rate - 0.4]

    keep_list = args.keep_ratio.split("_")
    keep_list_float = []
    for i in range(len(keep_list)):
        keep_list_float.append(float(keep_list[i]))

    # if args.sharing3[0] is not 0:
    sharing_list1, sharing_list2, sharing_list3 = args.sharing1.split("_"), args.sharing2.split("_"), args.sharing3.split("_")
    # else:
    #     sharing_list1, sharing_list2 = args.sharing1.split("_"), args.sharing2.split("_")

    layer_configs1, layer_configs2, layer_configs3 = [], [], []
    # print(f'###################################  length {len(sharing_list)} {sharing_list}')
    for i in range(len(sharing_list1)):
        layer_configs1.append(int(sharing_list1[i]))

    # print(sharing_list2[0])
    if int(sharing_list2[0]) != 0 and (sharing_list2[0]) != '100':
        for i in range(len(sharing_list2)):
            layer_configs2.append(int(sharing_list2[i]))
    else:
        layer_configs2 = []

    if int(sharing_list3[0]) != 0 and (sharing_list3[0]) != '100':
        for i in range(len(sharing_list3)):
            layer_configs3.append(int(sharing_list3[i]))
    else:
        layer_configs3 = []

    # if sharing_list3[0] is not 'X':
    #     for i in range(len(sharing_list3)):
    #         layer_configs3.append(int(sharing_list3[i]))
    # else:
    #     layer_configs3 = []


    print(f"Creating model: {args.model}")

    if args.model == 'deit-s':
        PRUNING_LOC = [3,6,9]
        # layer_configs = layer_configs
        KEEP_RATE = keep_list_float #args.keep_ratio
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        model1 = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=True, layer_configs=layer_configs1)

        effort1 = torch.load(args.effort1, map_location='cpu')

        # KEEP_RATE1 = [0.7, 0.5, 0.35]
        model2 = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=True, layer_configs=layer_configs2)

        effort2 = torch.load(args.effort2, map_location='cpu')

        print(f' sharing3 {sharing_list3[0]}')
        if sharing_list3[0] != '100':
            model3 = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=True, layer_configs=layer_configs3)

            effort3 = torch.load(args.effort3, map_location='cpu')

    elif args.model == 'deit-256':
        PRUNING_LOC = [3, 6, 9]
        # layer_configs = layer_configs
        KEEP_RATE = keep_list_float  # args.keep_ratio
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        model1 = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=True, layer_configs=layer_configs1)

        effort1 = torch.load(args.effort1, map_location='cpu')

        if 'best' in effort1:
            max_accuracy = effort1['best']
            print('Previous best accuracy: %.2f' % max_accuracy)
            # print('Previous best ema accuracy: %.2f' % effort1['best_ema'])

        # KEEP_RATE1 = [0.7, 0.5, 0.35]
        model2 = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=True, layer_configs=layer_configs2)

        effort2 = torch.load(args.effort2, map_location='cpu')

        if 'best' in effort2:
            max_accuracy = effort2['best']
            print('Previous best accuracy2: %.2f' % max_accuracy)

        print(f' sharing3 {sharing_list3[0]}')
        if sharing_list3[0] != '100':
            model3 = VisionTransformerDiffPruning(
                patch_size=16, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
                pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=True, layer_configs=layer_configs3)

            effort3 = torch.load(args.effort3, map_location='cpu')

            # if 'best' in effort3:
            #     max_accuracy = effort3['best']
            #     print('Previous best accuracy: %.2f' % max_accuracy)

        # pretrained = torch.load('../pretrained/dynamic-vit_384_r0.7.pth', map_location='cpu')



        # if args.effort3 is not None:


        # effort_list = []

        # pretrained = torch.load('../pretrained/deit_small_patch16_224-cd65a155.pth', map_location='cpu')

        # model.load_state_dict(pretrained)
        # teacher_model = VisionTransformerTeacher(
        #     patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True)

    # if args.model == 'deit-s':
    #     PRUNING_LOC = [3,6,9]
    #     # layer_configs = layer_configs
    #     KEEP_RATE = keep_list_float #args.keep_ratio
    #     print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
    #     model = VisionTransformerDiffPruning(
    #         patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
    #         pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=True, layer_configs=layer_configs
    #     )
    #     pretrained_shared = torch.load(args.model_path, map_location='cpu')
    #
    #     utils.load_state_dict(model, pretrained_shared)

    # elif args.model == 'deit-256':
    #     PRUNING_LOC = [3,6,9]
    #     print('token_ratio =', KEEP_RATE1, 'at layer', PRUNING_LOC)
    #     model = VisionTransformerDiffPruning(
    #         patch_size=16, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
    #         pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE1
    #         )
    elif args.model == 'deit-b':
        PRUNING_LOC = [4,8,12]
        print('token_ratio =', KEEP_RATE1, 'at layer', PRUNING_LOC)
        model = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE1
            )
    elif args.model == 'lvvit-s':
        PRUNING_LOC = [4, 8, 12]
        KEEP_RATE = keep_list_float  # args.keep_ratio
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        model1 = LVViTDiffPruning(
            patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
            p_emb='4_2', skip_lam=2., return_dense=True, mix_token=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=True, layer_configs=layer_configs1)

        effort1 = torch.load(args.effort1, map_location='cpu')

        # KEEP_RATE1 = [0.7, 0.5, 0.35]
        model2 = LVViTDiffPruning(
            patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
            p_emb='4_2', skip_lam=2., return_dense=True, mix_token=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=True, layer_configs=layer_configs2)

        effort2 = torch.load(args.effort2, map_location='cpu')

        print(f' sharing3 {sharing_list3[0]}')
        if sharing_list3[0] != '100':
            model3 = LVViTDiffPruning(
            patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
            p_emb='4_2', skip_lam=2., return_dense=True, mix_token=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=True, layer_configs=layer_configs3)

            effort3 = torch.load(args.effort3, map_location='cpu')


        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)

        # model = LVViTDiffPruning(
        #     patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
        #     p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True,
        #     pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE1
        # )
    elif args.model == 'lvvit-m':
        PRUNING_LOC = [5,10,15]
        print('token_ratio =', KEEP_RATE1, 'at layer', PRUNING_LOC)
        model = LVViTDiffPruning(
            patch_size=16, embed_dim=512, depth=20, num_heads=8, mlp_ratio=3.,
            p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE1
        )
    elif args.model == 'convnext-t':
        PRUNING_LOC = [1,2,3]
        print('token_ratio =', KEEP_RATE2, 'at layer', PRUNING_LOC)
        model = AdaConvNeXt(
            sparse_ratio=KEEP_RATE2, pruning_loc=PRUNING_LOC
        )
    elif args.model == 'convnext-s':
        PRUNING_LOC = [3,6,9]
        print('token_ratio =', KEEP_RATE2, 'at layer', PRUNING_LOC)
        model = AdaConvNeXt(
            sparse_ratio=KEEP_RATE2, pruning_loc=PRUNING_LOC,
            depths=[3, 3, 27, 3]
        )
    elif args.model == 'convnext-b':
        PRUNING_LOC = [3,6,9]
        print('token_ratio =', KEEP_RATE2, 'at layer', PRUNING_LOC)
        model = AdaConvNeXt(
            sparse_ratio=KEEP_RATE2, pruning_loc=PRUNING_LOC,
            depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]
        )
    elif args.model == 'swin-t':
        PRUNING_LOC = [1,2,3]
        print('token_ratio =', KEEP_RATE2, 'at layer', PRUNING_LOC)
        model = AdaSwinTransformer(
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_rate=0.0,
            pruning_loc=[1,2,3], sparse_ratio=KEEP_RATE2
        )
    elif args.model == 'swin-s':
        PRUNING_LOC = [2,4,6]
        print('token_ratio =', KEEP_RATE2, 'at layer', PRUNING_LOC)
        model = AdaSwinTransformer(
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_rate=0.0,
            drop_path_rate=args.drop_path,
            pruning_loc=[2,4,6], sparse_ratio=KEEP_RATE2
        )
    elif args.model == 'swin-b':
        PRUNING_LOC = [2,4,6]
        print('token_ratio =', KEEP_RATE2, 'at layer', PRUNING_LOC)
        model = AdaSwinTransformer(
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_rate=0.0,
            drop_path_rate=args.drop_path,
            pruning_loc=[2,4,6], sparse_ratio=KEEP_RATE2
        )
    else:
        raise NotImplementedError

    # model_path = args.model_path
    # # model_path = './logs/X-X-X-X-Q-X-X-Q-X-X-Q-X/checkpoint-best-ema.pth'
    # checkpoint = torch.load(model_path, map_location="cpu")
    # print(args.sharing2, type(args.sharing2), 'lvvit' in args.model, args.sharing2 == 0 )
    # if int(args.sharing2) == 0 and 'lvvit' in args.model:
    #
    #
    #     model1.load_state_dict(effort1["model"])
    #     model1 = ModelEma(model1, decay=0.9999, device='cuda', resume='')
    #     # best_acc1 = effort1["best"]
    #     # utils.load_state_dict(model2, effort2["model"])
    #     print(f'effort1 {effort1.keys()}')
    #     utils.load_state_dict(model2, effort2)
    #     # model2.load_state_dict(effort2)
    #     model2 = ModelEma(model2, decay=0.9999, device='cuda', resume='')
    #     # best_acc2 = effort2["best"]
    #     print(f'effort2 {effort2.keys()}')
    #     if sharing_list3[0] != '100':
    #         model3.load_state_dict(effort3)
    #         model3 = ModelEma(model3, decay=0.9999, device='cuda', resume='')
    # else:

        # print(f'else printed')
        # print(f'effort1 {effort2.keys()}')
    model1.load_state_dict(effort1["model_ema"])
    model1 = ModelEma(model1, decay=0.9999, device='cuda', resume='')
    # best_acc1 = effort1["best"]
    # utils.load_state_dict(model2, effort2["model"])

    model2.load_state_dict(effort2["model"])
    model2 = ModelEma(model2, decay=0.9999, device='cuda', resume='')
    # best_acc2 = effort2["best"]

    if sharing_list3[0] != '100':
        model3.load_state_dict(effort3["model"])
        model3 = ModelEma(model3, decay=0.9999, device='cuda', resume='')
        # best_acc3 = effort3["best"]

    print(effort1.keys())
    keys = list(effort1.keys())


    # print('## model has been successfully loaded')
    # print(f'acc1 {effort1[keys[5]]} acc2 {effort2[keys[5]]} acc3 {effort3[keys[5]]}')
    # model = model.cuda()

    # n_parameters = sum(p.numel() for p in model.parameters())
    # print('number of params:', n_parameters)

    criterion = torch.nn.CrossEntropyLoss().cuda()

    latency_effort1 = 1 #compute_latency_atr(base_rate=1, reused_encoders=layer_configs1)
    latency_effort2 = 2 #compute_latency_atr(base_rate=1, reused_encoders=layer_configs2) + latency_effort1
    if sharing_list3[0] != '100':
        latency_effort3 = 3 #compute_latency_atr(base_rate=1,
                             #                 reused_encoders=layer_configs3) + latency_effort1 + latency_effort2



    if sharing_list3[0] != '100':
        latency_tensor = torch.tensor([latency_effort1, latency_effort2, latency_effort3]).cuda()
        print(f'latency tensor {latency_tensor}')
        find_best_threshold(data_loader_val, [model1.ema, model2.ema, model3.ema], latency_tensor)
    else:
        latency_tensor = torch.tensor([latency_effort1, latency_effort2]).cuda()
        print(f'latency tensor {latency_tensor}')
        find_best_threshold(data_loader_val, [model1.ema, model2.ema], latency_tensor)


def find_best_threshold(dataloader, models, latency_tensor):
    th1 = np.linspace(0.001, 0.004, num=10)
    # th2 = np.array([0.0026666666666666666])
    th2 = np.linspace(0.0001, 0.0015, num=20)
    # th2 = [0]
    latency_list, acc_list = [], []
    max_acc = 0
    min_latency = 100
    if len(models) == 3:
        for threshold1 in th1:
            for threshold2 in th2:
                thresholds = [threshold1, threshold2, 1]
                latency, acc = validate(dataloader, models, latency_tensor, thresholds)
                print(thresholds, latency, acc)
                latency_list.append(latency)
                acc_list.append(acc)

                if acc > max_acc:
                    max_acc = acc
                    max_acc_latency = latency
                    threshold_acc = thresholds
                if latency < min_latency:
                    min_latency = latency
                    min_latency_acc = acc
                    threshold_lat = thresholds
    else:
        for threshold1 in th2:
            thresholds = [threshold1, 1]
            latency, acc = validate(dataloader, models, latency_tensor, thresholds)
            print(thresholds, latency, acc)
            latency_list.append(latency)
            acc_list.append(acc)
            if acc > max_acc:
                max_acc = acc
                max_acc_latency = latency
                threshold_acc = thresholds
            if latency < min_latency:
                min_latency = latency
                min_latency_acc = acc
                threshold_lat = thresholds

    print(f'best acc thresholds {threshold_acc} acc {max_acc} latency {max_acc_latency}')
    print(f'best latency thresholds {threshold_lat} acc {min_latency_acc} latency {min_latency}')









def validate(val_loader, models, latency_tensor, thresholds):

    # thresholds = [0.0025, 0.003, 1]
    total_acc = 0
    classified_in_effort = torch.FloatTensor([0]*len(models)).cuda()
    classified_per_batch = []
    n_images = 0
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            for i, (images, target) in (enumerate(val_loader)):
                # print(f'batch {i}')
                mask_not_classified_here = torch.ones(images.size(0)).cuda()
                for idx, model in enumerate(models):
                    # print(idx)
                    # model = model.cuda()
                    model.eval()
                    # start_time = time.time()
                    images = images.float().cuda()
                    target = target.cuda()
                    # end_time = time.time()
                    # execution_time = end_time - start_time
                    # print("loading time:", execution_time, "seconds")

                    # compute output
                    # start_time = time.time()
                    output = model(images)
                    # end_time = time.time()
                    # execution_time = end_time - start_time
                    # print(f"execution time: {idx} {execution_time}")

                    softmax_output = output.softmax(dim=-1)
                    entropy = -1 * (softmax_output * torch.log(softmax_output)).mean(dim=-1)
                    threshold = thresholds[idx]

                    mask_classified_here = entropy <= threshold

                    acc = mask_not_classified_here * mask_classified_here * (output.max(-1)[-1] == target)
                    classified = (mask_not_classified_here * mask_classified_here).sum()
                    total_acc += acc.sum()

                    classified_in_effort[idx] += classified
                    # classified_list.append(classified.cpu().numpy())

                    # print((mask_not_classified_here * mask_classified_here).sum())
                    mask_not_classified_here = (entropy > threshold) * mask_not_classified_here

                # print(classified_in_effort)
                n_images += images.size(0)
            print(f'classified_in_effort {classified_in_effort/n_images}')

            weighted_latency = classified_in_effort/n_images * latency_tensor
            # print(f'percent classified {classified_in_effort/n_images}')
            # print(f'weighted latency {weighted_latency}')
            # print(f'accuracy {total_acc/n_images} n_images {n_images}')

    return weighted_latency.sum(), total_acc/n_images

                    # acc_list.append(acc.sum())
                # print(acc_list, classified_list, total_classified)
                # classified_per_batch.append(classified_list)

            # classified_per_batch = np.array(classified_per_batch)
            # classified_per_batch = np.transpose(classified_per_batch)
            # print(np.shape(classified_per_batch))
            # for idx, _ in enumerate(models):
            #     for i
            #     effort1_classified += classified_per_batch[]


                    # measure accuracy and record loss
                    # acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    # losses.update(loss.item(), images.size(0))
                    # top1.update(acc1[0], images.size(0))
                    # top5.update(acc5[0], images.size(0))
                    #
                    # # measure elapsed time
                    # batch_time.update(time.time() - end)
                    # end = time.time()
                    #
                    # if i % 20 == 0:
                    #     progress.display(i)

                # TODO: this should also be done with the ProgressMeter
                # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                #       .format(top1=top1, top5=top5))


    # return top1.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dynamic evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
