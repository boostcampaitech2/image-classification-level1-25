import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
from albumentations.augmentations.geometric.functional import resize

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torchvision import transforms

from datasets.dataset import MaskBaseDataset
from module.loss import create_criterion

import timm
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from sklearn.metrics import f1_score
from tqdm import tqdm
import pandas as pd

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{save_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def train(args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(args.save_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- augmentation
    train_transform_module = getattr(import_module("trans." + args.usertrans), args.trainaug)  # default: BaseAugmentation
    train_transform = train_transform_module(
        resize=args.resize,
    )
    valid_transform_module = getattr(import_module("trans." + args.usertrans), args.validaug)  # default: BaseAugmentation
    valid_transform = valid_transform_module(
        resize=args.resize,
    )

    # -- dataset
    dataset_module = getattr(import_module("datasets." + args.userdataset), args.traindataset)  # default: BaseAugmentation
    train_dataset = dataset_module(
        data_dir=args.data_dir,
        val_ratio=args.val_ratio,
        mode='train',
        transform = train_transform
    )
    valid_dataset = dataset_module(
        data_dir=args.data_dir,
        val_ratio=args.val_ratio,
        mode='valid',
        transform = valid_transform
    )

    # -- data_loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        # num_workers=0,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        # num_workers=0,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- multi-model setting
    num_classes_dict = train_dataset.num_classes # dictionary types, keys : ['mask', 'gender', 'age', 'concat', 'merged'], values : [3, 2, 3, 8, 18]
    labels_classes = ['mask', 'gender', 'age']
    model_dict = {}
    optimizer_dict = {}
    scheduler_dict = {}
    
    # -- model
    ## -- multi model
    for idx, label_class in enumerate(labels_classes) :
        model_module = getattr(import_module("models."+args.usermodel), args.models[idx])  # default: resnetbase
        model_dict[label_class] = model_module(
            num_classes=num_classes_dict[label_class]
        ).to(device)
        # model_dict[label_class] = torch.nn.DataParallel(model_dict[label_class])


    # -- loss & metric
    for label_class in labels_classes :
        criterion = create_criterion(args.criterion)  # default: cross_entropy
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
        optimizer_dict[label_class] = opt_module(
            filter(lambda p: p.requires_grad, model_dict[label_class].parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )
        scheduler_dict[label_class] = StepLR(optimizer_dict[label_class], args.lr_decay_step, gamma=0.5)

    # -- logging
    # logger = SummaryWriter(log_dir=save_dir)
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    except OSError:
        print ('Error: Creating directory. ' +  save_dir)

    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    
    for idx, split_list in enumerate(labels_classes) : ## MM model
        print(f'-'*50)
        print(split_list)

        best_val_loss = np.inf
        best_val_acc = 0
        best_val_f1 = 0

        num_epoch = int(args.epochs[idx])
        for epoch in range(num_epoch):
            # train loop
            model_dict[split_list].train() ## MM model
            loss_value = 0
            matches = 0
            f1_sum = 0
            print(f"Epoch[{epoch}/{num_epoch}]")

            for idx, train_batch in enumerate(pbar := tqdm(train_loader, ncols=100)):
                inputs, labels_dict = train_batch
                inputs = inputs.to(device)
                labels = labels_dict[split_list].to(device)  ## MM model

                optimizer_dict[split_list].zero_grad() ## MM model

                outs = model_dict[split_list](inputs)   ## MM model
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

                loss.backward()
                optimizer_dict[split_list].step()  ## MM model

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                f1_sum += f1_score(labels.data.cpu().numpy(), preds.cpu().numpy(), average='macro')
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / (idx+1)
                    train_acc = matches / args.batch_size / (idx+1)
                    train_f1 = f1_sum / (idx+1)
                    current_lr = get_lr(optimizer_dict[split_list])
                    pbar.set_description(f"loss_{train_loss:4.4}, f1_{train_f1:4.4}, acc_{train_acc:4.2%}, lr_{current_lr}")

            scheduler_dict[split_list].step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model_dict[split_list].eval()   ## MM model
                val_loss_items = []
                val_acc_items = []
                val_f1_items = []
                figure = None
                for val_batch in tqdm(val_loader, ncols=100):
                    inputs, labels_dict = val_batch
                    inputs = inputs.to(device)
                    labels = labels_dict[split_list].to(device) ## MM model

                    outs = model_dict[split_list](inputs)   ## MM model
                    preds = torch.argmax(outs, dim=-1)

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    f1 = f1_score(labels.data.cpu().numpy(), preds.cpu().numpy(), average='macro')
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)
                    val_f1_items.append(f1)

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(valid_dataset)
                val_f1 = np.sum(val_f1_items) / len(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    
                if val_acc > best_val_acc:
                #     print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                #     torch.save(model.state_dict(), f"{save_dir}/best.pth")
                    best_val_acc = val_acc
                    print(f"New best model for val f1 score : {val_f1:.4}! saving the best model..")
                if val_f1 > best_val_f1 :
                    torch.save(model_dict[label_class].state_dict(), f"{save_dir}/{split_list}_best.pth")     ## MM model
                    best_val_f1 = val_f1
                torch.save(model_dict[label_class].state_dict(), f"{save_dir}/{split_list}_last.pth")     ## MM model
                print(
                    f"[Val] acc : {val_acc:4.2%}, f1 : {val_f1:.4f}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best f1: {best_val_f1:.4f}, best loss: {best_val_loss:4.2}"
                )

                print()

    print(f'-'*50)
    print('merged_model')
    # ## -- merged mergedmodel
    model_module = getattr(import_module("models."+args.usermodel), args.mergedmodel)  # default: MultiModelMergeModel
    merged_model = model_module(
        model_dict['mask'], model_dict['gender'], model_dict['age'], 
        concatclasses=num_classes_dict['concat'], num_classes=num_classes_dict['merged'],
        prev_model_frz=args.prev_model_frz
    ).to(device)

    best_val_loss = np.inf
    best_val_acc = 0
    best_val_f1 = 0
    
    if args.mergedmodel == 'MultiModelMergeModel' : ## model test
        criterion = create_criterion(args.criterion)  # default: cross_entropy
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
        merged_optimizer = opt_module(
            filter(lambda p: p.requires_grad, merged_model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )
        merged_scheduler = StepLR(merged_optimizer, args.lr_decay_step, gamma=0.5)
        
        num_epoch = int(args.epochs[-1])

        for epoch in range(num_epoch):

                # train loop
                merged_model.train() ## MM model
                loss_value = 0
                matches = 0
                f1_sum = 0
                print(f"Epoch[{epoch}/{num_epoch}]")

                for idx, train_batch in enumerate(pbar := tqdm(train_loader, ncols=100)):
                    inputs, labels_dict = train_batch
                    inputs = inputs.to(device)
                    labels = labels_dict['merged'].to(device)  ## MM model

                    merged_optimizer.zero_grad() ## MM model

                    outs = merged_model(inputs)   ## MM model
                    preds = torch.argmax(outs, dim=-1)
                    loss = criterion(outs, labels)

                    loss.backward()
                    merged_optimizer.step()  ## MM model

                    loss_value += loss.item()
                    matches += (preds == labels).sum().item()
                    f1_sum += f1_score(labels.data.cpu().numpy(), preds.cpu().numpy(), average='macro')
                    if (idx + 1) % args.log_interval == 0:
                        train_loss = loss_value / (idx+1)
                        train_acc = matches / args.batch_size / (idx+1)
                        train_f1 = f1_sum / (idx+1)
                        current_lr = get_lr(merged_optimizer)
                        pbar.set_description(f"loss_{train_loss:4.4}, f1_{train_f1:4.4}, acc_{train_acc:4.2%}, lr_{current_lr}")
                merged_scheduler.step()

                # val loop
                with torch.no_grad():
                    print("Calculating validation results...")
                    merged_model.eval()   ## MM model
                    val_loss_items = []
                    val_acc_items = []
                    val_f1_items = []
                    figure = None
                    for val_batch in tqdm(val_loader, ncols=100):
                        inputs, labels_dict = val_batch
                        inputs = inputs.to(device)
                        labels = labels_dict['merged'].to(device) ## MM model

                        outs = merged_model(inputs)   ## MM model
                        preds = torch.argmax(outs, dim=-1)

                        loss_item = criterion(outs, labels).item()
                        acc_item = (labels == preds).sum().item()
                        f1 = f1_score(labels.data.cpu().numpy(), preds.cpu().numpy(), average='macro')
                        val_loss_items.append(loss_item)
                        val_acc_items.append(acc_item)
                        val_f1_items.append(f1)

                    val_loss = np.sum(val_loss_items) / len(val_loader)
                    val_acc = np.sum(val_acc_items) / len(valid_dataset)
                    val_f1 = np.sum(val_f1_items) / len(val_loader)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        print(f"New best model for val f1 score : {val_f1:.4}! saving the best model..")
                    if val_f1 > best_val_f1 :
                        torch.save(merged_model.state_dict(), f"{save_dir}/{split_list}_best.pth")     ## MM model
                        best_val_f1 = val_f1
                    torch.save(merged_model.state_dict(), f"{save_dir}/{split_list}_last.pth")     ## MM model
                    print(
                        f"[Val] acc : {val_acc:4.2%}, f1 : {val_f1:.4f}, loss: {val_loss:4.2} || "
                        f"best acc : {best_val_acc:4.2%}, best f1: {best_val_f1:.4f}, best loss: {best_val_loss:4.2}"
                    )
                    print()
    else :  ## model test
        num_epoch = int(args.epochs[-1])
        for epoch in range(num_epoch):
            with torch.no_grad():
                print("Calculating validation results...")
                merged_model.eval()   ## MM model
                val_loss_items = []
                val_acc_items = []
                val_f1_items = []

                for val_batch in tqdm(val_loader, ncols=100):
                    inputs, labels_dict = val_batch
                    inputs = inputs.to(device)
                    labels = labels_dict['merged'].to(device) ## MM model
                    
                    outs = merged_model(inputs)   ## MM model
                    preds = torch.argmax(outs, dim=-1)
                    
                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    f1 = f1_score(labels.data.cpu().numpy(), preds.cpu().numpy(), average='macro')
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)
                    val_f1_items.append(f1)

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(valid_dataset)
                val_f1 = np.sum(val_f1_items) / len(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    print(f"New best model for val f1 score : {val_f1:.4}! saving the best model..")
                if val_f1 > best_val_f1 :
                    torch.save(merged_model.state_dict(), f"{save_dir}/{split_list}_best.pth")     ## MM model
                    best_val_f1 = val_f1
                torch.save(merged_model.state_dict(), f"{save_dir}/{split_list}_last.pth")     ## MM model
                print(
                    f"[Val] acc : {val_acc:4.2%}, f1 : {val_f1:.4f}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best f1: {best_val_f1:.4f}, best loss: {best_val_loss:4.2}"
                )
                print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # from dotenv import load_dotenv
    import os
    # load_dotenv(verbose=True)1

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=25, help='random seed (default: 25)')
    parser.add_argument('--epochs', nargs="+", type=int, default=[1,1,1,1], help='number of epochs to train (default: [1,1,1,1])')
    parser.add_argument('--traindataset', type=str, default='MMteamDataset', help='train dataset augmentation type (default: MMteamDataset)')
    parser.add_argument('--validdataset', type=str, default='MMteamDataset', help='validation dataset augmentation type (default: MMteamDataset)')
    parser.add_argument('--trainaug', type=str, default='A_simple_trans', help='data augmentation type (default: A_simple_trans)')
    parser.add_argument('--validaug', type=str, default='A_centercrop_trans', help='data augmentation type (default: A_centercrop_trans)')
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224], help='resize size for image when training (default: [224,224])')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--valid_batch_size', type=int, default=32, help='input batch size for validing (default: 1000)')
    parser.add_argument('--models', nargs="+", type=str, default=['resnetbase','resnetbase','resnetbase'], help='model type (default: resnetbase, resnetbase, resnetbase)')
    parser.add_argument('--mergedmodel', type=str, default='MultiModelMergeModel', help='model type (default: MultiModelMergeModel)')
    parser.add_argument('--prev_model_frz', type=str, default='True', help='True/False (default: True)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='ratio for validaton (default: 0.1)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--patience',type=int, default = 5, help = 'earlystopping rounds')
    parser.add_argument('--name', default='exp', help='model save at {SM_SAVE_DIR}/{name}')
    parser.add_argument('--userdataset', default='dataset', help='select user custom dataset')
    parser.add_argument('--usermodel', default='model', help='select user custom model')
    parser.add_argument('--usertrans', default='trans', help='select user custom transform')
    
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/'))
    parser.add_argument('--save_dir', type=str, default=os.environ.get('SM_SAVE_DIR', './save'))

    args = parser.parse_args()

    train(args)