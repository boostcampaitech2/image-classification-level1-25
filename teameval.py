import argparse
import os
from importlib import import_module

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from datasets.dataset import basicDatasetA, MaskBaseDataset


def load_model(saved_model, filename, modelname, num_classes, device):
    model = None
    if len(data := [s for s in os.listdir(saved_model) if s.endswith(filename)]) == 1:
        model_cls = getattr(import_module("models.model"), modelname)
        model = model_cls(
            num_classes=num_classes
        )
        # model = torch.nn.DataParallel(model)

        model_path = os.path.join(saved_model, data[0])
        model.load_state_dict(torch.load(model_path, map_location=device))
    else :
        model_cls = getattr(import_module("models.model"), 'ensemble')
        model = model_cls(
            modelname = modelname,
            length = len(data),
            num_classes=num_classes,
            device = device
        )
        # model = torch.nn.DataParallel(model)

        for M, d in zip(model.superM, data):
            model_path = os.path.join(saved_model, d)
            M.load_state_dict(torch.load(model_path, map_location=device))
    return model


@torch.no_grad()
def inference(args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(args.save_dir, args.filename, args.model, args.num_classes, device).to(device)

    model.eval()

    valid_transform_module = getattr(import_module("trans.trans"), args.validaug)  # default: BaseAugmentation
    valid_transform = valid_transform_module(
        resize=args.resize,
    )

    dataset = basicDatasetA(data_dir=args.data_dir, mode='eval', transform=valid_transform)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    val_acc_items = []
    val_f1_items = []
    with torch.no_grad():
        for idx, train_batch in enumerate(loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)

            acc_item = (labels == preds).sum().item()
            f1 = f1_score(labels.data.cpu().numpy(), preds.cpu().numpy(), average='macro')
            val_acc_items.append(acc_item)
            val_f1_items.append(f1)
    
        val_acc = np.sum(val_acc_items) / len(dataset)
        val_f1 = np.sum(val_f1_items) / len(loader)
        print(
                f"[Eval] acc : {val_acc:4.2%}, f1: {val_f1:5.4}"
            )

    print(f'Team eval Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for validing (default: 128)')
    parser.add_argument('--model', type=str, default='rexnet_200base', help='model type (default: BaseModel)')
    parser.add_argument('--filename', type=str, default='best.pth', help='save file name (default: best.pth)')
    parser.add_argument('--validaug', type=str, default='A_centercrop_trans', help='validation data augmentation type (default: A_centercrop_trans)')
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224], help='resize size for image when training')
    parser.add_argument('--num_classes',type=int, default = 18, help = 'num_classes')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/train/'))
    parser.add_argument('--save_dir', type=str, default=os.environ.get('SM_CHANNEL_SAVE', './save'))

    args = parser.parse_args()

    inference(args)

