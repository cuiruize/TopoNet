import argparse
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import evaluation
from utils import prepare_dataset
from utils.dataset import LandmarkDataset, save_img
from models.TopoNet import TopoNet
from medpy.metric import assd
import numpy as np



def main(args):
    train_file, test_file, val_file = prepare_dataset.get_split(args.data_path)

    test_dataset = LandmarkDataset(test_file, transform=None, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda")

    model = TopoNet(1024, 1024, depth_path=args.depth_path).to(device)
    model_checkpoint = torch.load(args.model_path)
    model.load_state_dict(model_checkpoint)
    model.eval()

    validation_IOU = []
    mDice = []
    mAssd = []

    for index, (X_batch, depth, y_batch, name) in tqdm(enumerate(test_loader)):

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        depth = depth.to(device)

        output, _ = model(X_batch)

        output = torch.argmax(torch.softmax(output, dim=1), dim=1).detach().cpu().numpy()

        y_batch = torch.argmax(y_batch, dim=1)

        tmp2 = y_batch.detach().cpu().numpy()
        tmp = output
        tmp = tmp[0]
        tmp2 = tmp2[0]

        pred = np.array([tmp == i for i in range(4)]).astype(np.uint8)
        gt = np.array([tmp2 == i for i in range(4)]).astype(np.uint8)

        iou, dice = evaluation(pred[1:].flatten(), gt[1:].flatten())

        validation_IOU.append(iou)
        mDice.append(dice)

        if 0 == np.count_nonzero(pred[1:]):
            mAssd.append(80)
        else:
            assd_value = assd(pred[1:], gt[1:])
            mAssd.append(assd_value)

        toprint = save_img(tmp)
        cv2.imwrite(args.save_path + str(name).split('/', 6)[-1].replace('/', '_')[:-3], toprint)


    print(np.mean(validation_IOU))
    print(np.mean(mDice))
    print(np.mean(mAssd))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',default="best_model_path.pth")
    parser.add_argument('--data_path', default='data/')
    parser.add_argument('--depth_path', default='depth_anything_v2_vitb.pth')
    parser.add_argument('--save_path', default='test_results/')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    main(args=args)
