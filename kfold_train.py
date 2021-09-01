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
from torch.utils.tensorboard import SummaryWriter
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
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
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


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))
    test_dir = args.eval_dir
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("datasets." + args.userdataset), args.dataset)  # default: team
    all_dataset = dataset_module(
        data_path=data_dir,
        train = 'ALL'
    )
    num_classes = all_dataset.num_classes

    # -- train dataset
    train_dataset = dataset_module(
        data_path=data_dir,
        train = 'train'
    )

    # -- team_eval_dataset
    team_eval_dataset = dataset_module(
        data_path=data_dir,
        train = 'eval'
    )

    # -- test_dataset
    test_dataset_module = getattr(import_module("datasets." + args.userdataset), "TestDataset")
    test_dataset = test_dataset_module(
        img_paths = image_paths,
        resize = args.resize
    )

    # -- augmentation
    train_transform_module = getattr(import_module("trans." + args.usertrans), args.trainaug)  # default: BaseAugmentation
    train_transform = train_transform_module(
        resize=args.resize,
    )
    valid_transform_module = getattr(import_module("trans." + args.usertrans), args.validaug)  # default: BaseAugmentation
    valid_transform = valid_transform_module(
        resize=args.resize,
    )

    all_dataset.set_transform(train_transform)
    team_eval_dataset.set_transform(train_transform)
    # test_dataset.set_transform(train_transform)


    # -- data_loader
    team_eval_loader = DataLoader(
        team_eval_dataset,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )


    team_eval_preds = [0 for _ in range(len(team_eval_dataset))]
    test_preds = [0 for _ in range(len(test_dataset))]
    skf = StratifiedKFold(n_splits=args.num_split, shuffle=True, random_state=25)
    for fold, (train_ids, valid_ids) in enumerate(skf.split(train_dataset.df_csv, train_dataset.df_csv.gender_age_cls)):
        # Print
        print('--------------------------------')
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        # -- Image index
        train_image_ids = sum([[x*7+i for i in range(7)] for x in train_ids],[])
        valid_image_ids = sum([[y*7+i for i in range(7)] for y in valid_ids],[])

        # -- Dataset
        train_dataset = Subset(all_dataset,train_image_ids)
        valid_dataset = Subset(all_dataset,valid_image_ids)

        print(f'train length : {len(train_dataset)}')
        print(f'valid length : {len(valid_dataset)}')

        # -- DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )
        print(f'train length : {len(train_loader)}')
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.valid_batch_size,
            shuffle=True,
            drop_last=True
        )
        print(f'valid length : {len(valid_loader)}')

        # -- model
        model_module = getattr(import_module("models."+args.usermodel), args.model)  # default: resnetbase
        model = model_module(
            num_classes=num_classes
        ).to(device)
        model = torch.nn.DataParallel(model)

        # -- loss & metric
        criterion = create_criterion(args.criterion)  # default: cross_entropy
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)


        counter = 0
        best_val_acc = 0
        best_val_loss = np.inf
        for epoch in range(args.epochs):
            print('*** Epoch {} ***'.format(epoch))

            # Training
            model.train()  
            running_loss, running_acc, running_f1 = 0.0, 0.0, 0.0
            
            # Set Trans
            all_dataset.set_transform(train_transform)

            for (inputs, labels) in tqdm(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(True):
                    logits = model(inputs)
                    _, preds = torch.max(logits, 1)
                    loss = criterion(logits, labels)

                    loss.backward()
                    optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.shape[0]
                    running_acc += torch.sum(preds == labels.data)
                    running_f1 += f1_score(labels.data.cpu().numpy(), preds.cpu().numpy(), average = 'macro')

            epoch_acc = running_acc/len(train_loader.dataset)
            epoch_loss = running_loss/len(train_loader.dataset)
            epoch_f1 = running_f1/len(train_loader)
            print('{} Loss: {:.4f} Acc: {:.4f} F1-score: {:.4f}'.format('train', epoch_loss, epoch_acc, epoch_f1))
        

            # Validation
            model.eval()
            valid_acc, valid_f1,valid_loss = 0.0, 0.0, 0.0
            
            # Set Trans
            all_dataset.set_transform(valid_transform)

            for val_batch in tqdm(valid_loader):
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(False):
                    logits = model(inputs)
                    _, preds = torch.max(logits, 1)

                    # statistics
                    valid_acc += torch.sum(preds == labels.data)
                    valid_loss += criterion(logits, labels).item()
                    valid_f1 += f1_score(labels.data.cpu().numpy(), preds.cpu().numpy(), average = 'macro')
                    

            valid_acc /= len(valid_loader.dataset)
            valid_f1 /= len(valid_loader)
            valid_loss /= len(valid_loader.dataset)

            best_val_loss = min(best_val_loss,valid_loss)
            if valid_acc > best_val_acc:
                print("New best model for val accuracy!")
                print(f"val_acc : {valid_acc}")
                best_val_acc = valid_acc
                counter = 0
                
            else:
                counter += 1
            # patience 횟수 동안 성능 향상이 없을 경우 학습을 종료시킵니다.
            if counter > args.patience:
                print("Early Stopping...")
                break


            print('{} Acc: {:.4f} f1-score: {:.4f}\n'.format('valid', valid_acc, valid_f1))

        # team_eval_pred
        all_predictions = []
        answers = []
        for images, labels in tqdm(team_eval_loader):
            with torch.no_grad():
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                all_predictions.extend(outputs.cpu().numpy())
                answers.extend(labels.cpu().numpy())
        team_eval_preds = [x+y for x,y in zip(team_eval_preds,all_predictions)]

        # test_pred
        all_predictions = []
        for images in tqdm(test_loader):
            with torch.no_grad():
                images = images.to(device)
                outputs = model(images)
                all_predictions.extend(outputs.cpu().numpy())

        test_preds = [x+y for x,y in zip(test_preds,all_predictions)]

    # Check Result
    print(f'Team eval accuracy : {torch.sum(torch.tensor(answers) == torch.tensor(np.argmax(team_eval_preds,axis=1)))/len(answers):.4}, f1-score : {f1_score(answers,np.argmax(team_eval_preds,axis=1),average="macro"):.4}')
    submission['ans'] = np.argmax(test_preds,axis = 1)
    submission.to_csv('stratified.csv', index=False)
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # from dotenv import load_dotenv
    import os
    # load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=25, help='random seed (default: 25)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='teamDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--trainaug', type=str, default='A_centercrop_trans', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--validaug', type=str, default='A_centercrop_trans', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=32, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='resnetbase', help='model type (default: resnetbase)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--patience',type=int, default = 5, help = 'earlystopping rounds')
    parser.add_argument('--num_split',type=int, default = 5, help = 'number of k-folds')
    parser.add_argument('--userdataset', default='dataset', help='select user custom dataset')
    parser.add_argument('--usermodel', default='model', help='select user custom model')
    parser.add_argument('--usertrans', default='trans', help='select user custom transform')
    

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--eval_dir',type=str, default= os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--save_dir', type=str, default=os.environ.get('SM_SAVE_DIR', './save'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)