import argparse
import os
from importlib import import_module

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.dataset import TestDataset, MaskBaseDataset

from tqdm import tqdm

def load_model(saved_model, modelname, num_classes, device):
    model_cls = getattr(import_module("models.model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )
    # model = torch.nn.DataParallel(model)
    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, modelname)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 18

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    # valid_transform_module = getattr(import_module("trans.trans"), args.validaug)  # default: BaseAugmentation
    # valid_transform = valid_transform_module(
    #     resize=args.resize,
    # )
    # dataset.set_transform(valid_transform)


    test_preds = [0 for _ in range(len(dataset))]
    print(os.listdir(args.save_dir))
    for saved_model in os.listdir(args.save_dir):
        if saved_model[-8:] == 'best.pth':
            model = load_model(args.save_dir, saved_model, num_classes, device).to(device)
            model.eval()
            all_predictions = []
            for images in tqdm(loader):
                with torch.no_grad():
                    images = images.to(device)
                    outputs = model(images)
                    all_predictions.extend(outputs.cpu().numpy())

            test_preds = [x+y for x,y in zip(test_preds,all_predictions)]


    info['ans'] = np.argmax(test_preds,axis=1)
    info.to_csv(os.path.join(args.save_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='rexnet_200base', help='model type (default: rexnet)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--save_dir', type=str, default=os.environ.get('SM_save_DATA_DIR', './save'))
    parser.add_argument('--validaug', type=str, default='A_centercrop_trans', help='data augmentation type (default: BaseAugmentation)')
    
    args = parser.parse_args()

    data_dir = args.data_dir

    # os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, args)


