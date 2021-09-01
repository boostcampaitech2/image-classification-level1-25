import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.dataset import TestDatasetA, MaskBaseDataset


def load_model(saved_model, filename, modelname, num_classes, device):
    model = None
    if len(data := [s for s in os.listdir(saved_model) if s.endswith(filename)]) == 0:
        raise Exception(f'cant find file. {filename}')
    elif len(data) == 1 :
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

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(args.save_dir, args.filename, args.model, num_classes, device).to(device)

    model.eval()

    img_root = os.path.join(args.data_dir, 'images')
    info_path = os.path.join(args.data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]

    valid_transform_module = getattr(import_module("trans.trans"), args.validaug)  # default: BaseAugmentation
    valid_transform = valid_transform_module(
        resize=args.resize,
    )

    dataset = TestDatasetA(img_paths, args.resize, transform=valid_transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(args.save_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for validing (default: 128)')
    parser.add_argument('--model', type=str, default='rexnet_200base', help='model type (default: BaseModel)')
    parser.add_argument('--filename', type=str, default='best.pth', help='save file name (default: best.pth)')
    parser.add_argument('--validaug', type=str, default='A_centercrop_trans', help='validation data augmentation type (default: A_centercrop_trans)')
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224], help='resize size for image when training')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--save_dir', type=str, default=os.environ.get('SM_CHANNEL_SAVE', './save'))

    args = parser.parse_args()

    inference(args)

