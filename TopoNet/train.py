import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from utils.metrics import evaluation, _dice_loss
from utils import prepare_dataset
from utils.dataset import LandmarkDataset
from models.TopoNet import TopoNet
from cldice.cldice import soft_dice_cldice
from utils.betti_loss import *
import surface_distance
from surface_distance import metrics


def main(args):

    train_file, test_file, val_file = prepare_dataset.get_split(args.data_path)
    train_dataset = LandmarkDataset(train_file, transform=None, mode='train')
    val_dataset = LandmarkDataset(test_file, transform=None, mode='val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device("cuda")
    cl_dice_loss = soft_dice_cldice(exclude_background=True)
    betti_loss = FastBettiMatchingLoss(
        filtration_type=FiltrationType.SUPERLEVEL,
        num_processes=16,
        convert_to_one_vs_rest=False,
        ignore_background=True,
        push_unmatched_to_1_0=True,
        barcode_length_threshold=0.1,
        topology_weights=[0.5, 0.5]
    )
    best_dice = -100

    model = TopoNet(1024, 1024, depth_path=args.depth_path).to(device)
    model.depth_encoder.requires_grad_(False)

    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, args.decay_lr)

    for epoch in range(args.epoch):
        epoch_running_loss = 0
        epoch_seg_loss = 0
        epoch_betti_loss = 0

        # training
        model.train()
        num_iterations = len(train_loader)
        for batch_idx, (X_batch, depth, y_batch, name) in tqdm(enumerate(train_loader)):

            optimizer.zero_grad()

            if X_batch.shape[0] == 1:
                continue

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            depth = depth.to(device)

            output, feature = model(X_batch)

            if epoch > 5:
                p = float(batch_idx + (epoch + 1) * num_iterations) / (200 * num_iterations)
                alpha = (2. / (1. + np.exp(-10 * p)) - 1) * 0.05 # betti loss warmup
                betti, _ = betti_loss(output, y_batch)
                seg_loss = cl_dice_loss(y_batch, output)
                loss = betti * alpha + seg_loss * (1 - alpha)
            else: # dice loss warmup
                seg_loss = _dice_loss(y_batch, output)
                loss = seg_loss
            
            loss.backward()

            optimizer.step()

            epoch_running_loss += loss.item()
            epoch_seg_loss += seg_loss.item()
            if epoch > 5:
                epoch_betti_loss += betti.item()
                

        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch, args.epoch, epoch_running_loss / (batch_idx + 1)))
        print('epoch [{}/{}], seg loss:{:.4f}'
              .format(epoch, args.epoch, epoch_seg_loss / (batch_idx + 1)))
        print('epoch [{}/{}], betti loss:{:.4f}'
              .format(epoch, args.epoch, epoch_betti_loss / (batch_idx + 1)))

        # validation
        model.eval()
        validation_IOU = []
        mDice = []
        mAssd = []

        with torch.no_grad():
            for X_batch, depth, y_batch, name in tqdm(val_loader):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                depth = depth.to(device)

                output, _ = model(X_batch)
                output = torch.argmax(torch.softmax(output, dim=1), dim=1)
                y_batch = torch.argmax(y_batch, dim=1)

                tmp2 = y_batch.detach().cpu().numpy()
                tmp = output.detach().cpu().numpy()
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
                    temp_assd = []
                    for i in range(3):
                        surface_distances = metrics.compute_surface_distances(np.array(gt[i + 1], dtype=bool),
                                                                              np.array(pred[i + 1], dtype=bool),
                                                                              (1.0, 1.0))
                        assd_value = surface_distance.compute_average_surface_distance(surface_distances)
                        temp_assd.append(assd_value[1])
                    if np.mean(temp_assd) < 500:
                        mAssd.append(np.mean(temp_assd))
                    else:
                        mAssd.append(80)

        print(np.mean(validation_IOU))
        print(np.mean(mDice))
        print(np.mean(mAssd))
        if np.mean(mDice) > best_dice:
            best_dice = np.mean(mDice)
            torch.save(model.state_dict(), args.save_path + "best_model_path.pth")
        print("best dice is:{:.4f}".format(best_dice))
    scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--weight_decay', default=3e-5)
    parser.add_argument('--decay_lr', default=1e-6)
    parser.add_argument('--epoch', default=200)
    parser.add_argument('--data_path', default='data/')
    parser.add_argument('--depth_path', default='depth_anything_v2_vitb.pth')
    parser.add_argument('--save_path', default='train_results/')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    main(args=args)
