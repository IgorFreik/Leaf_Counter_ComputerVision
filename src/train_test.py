import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


def loss_function(y_real, y_pred):
    # dice loss
    num = (y_real * y_pred).view(4, -1).sum(1)
    den = (y_real + y_pred).view(4, -1).sum(1)
    res = -2 * num / den
    dice = res.mean()

    # bce loss
    # temp = (-0.5 * y_real * torch.log(y_pred))
    # temp_sum = torch.sum(temp, (1, 2, 3))
    # bce = temp_sum.mean()
    bce = 0.5 * nn.BCELoss()(y_pred, y_real)
    # print('Dice:', dice, 'BCE:', bce)
    return dice + bce


def train(model, epochs, opt_segm, opt_count, data_tr, data_val, device):
    train_segm_losses = []
    train_count_losses = []
    val_segm_losses = []
    val_count_losses = []

    for epoch in range(epochs):
        avg_segm_loss = 0
        avg_count_loss = 0
        model.segmenter.train()
        model.counter.train()

        print('Training')

        for imgs, counts, masks in tqdm(data_tr):
            imgs, counts, masks = imgs.to(device), counts.to(device), masks.to(device)

            # Train segmentation
            opt_segm.zero_grad()
            output = model.segmenter(imgs)
            loss = loss_function(masks, output)
            loss.backward()
            opt_segm.step()
            avg_segm_loss += loss.item()

            # Train counting
            opt_count.zero_grad()
            output = model.counter(torch.cat((imgs, masks), dim=1)).squeeze(1)
            loss = nn.MSELoss()(output, counts.float())
            loss.backward()
            opt_count.step()
            avg_count_loss += loss.item()

        train_segm_losses.append(avg_segm_loss / len(data_tr))
        train_count_losses.append(avg_count_loss / len(data_tr))

        print('Evaluating')

        model.segmenter.eval()
        model.counter.eval()
        with torch.no_grad():
            avg_segm_loss = 0
            avg_count_loss = 0

            for imgs, counts, masks in tqdm(data_val):
                imgs, counts, masks = imgs.to(device), counts.to(device), masks.to(device)
                # Evaluate segmentation
                output_segm = model.segmenter(imgs)
                loss = loss_function(masks, output_segm)
                avg_segm_loss += loss.item()

                # Evaluate counts
                output_count = model.counter(torch.cat((imgs, masks), dim=1)).squeeze(1)
                loss = nn.MSELoss()(output_count, counts.float())
                avg_count_loss += loss.item()

            val_segm_losses.append(avg_segm_loss / len(data_val))
            val_count_losses.append(avg_count_loss / len(data_val))

            # Visualize tools
            clear_output(wait=True)
            batch_len = len(imgs)

            fig1, axes = plt.subplots(2, batch_len, figsize=(10, 6))
            for k in range(batch_len):
                np_img_orig = np.rollaxis(imgs[k].detach().cpu().numpy(), 0, 3)
                axes[0, k].imshow(np_img_orig, cmap='gray')
                axes[0, k].set_title(f'Real: {counts[k]}')
                axes[0, k].axis('off')

                np_img_segm = output_segm[k].squeeze(0).cpu().numpy()
                axes[1, k].imshow(np_img_segm, cmap='gray')
                axes[1, k].set_title(f'Pred: {output_count[k]:.1f}')
                axes[1, k].axis('off')

            fig1.suptitle(
                f'{epoch + 1} / {epochs} - Segm loss: {val_segm_losses[-1] :.2f}, Count loss: {val_count_losses[-1] :.2f}')

            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.plot(range(len(val_count_losses)), val_count_losses, label='Test')
            ax1.plot(range(len(train_count_losses)), train_count_losses, label='Train')
            ax1.legend()

            ax2.plot(range(len(val_segm_losses)), val_segm_losses, label='Test')
            ax2.plot(range(len(train_segm_losses)), train_segm_losses, label='Train')

            ax1.set_title('Counting')
            ax2.set_title('Segmentation')
            ax2.legend()
            plt.show()

    return {'train_segm': train_segm_losses, 'train_count': train_count_losses, 'val_segm': val_segm_losses,
            'val_count': val_count_losses}


# Evaluation functions to be added
def iou(outputs: torch.Tensor, labels: torch.Tensor):
    outputs = outputs.squeeze(1).byte()  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1).byte()
    SMOOTH = 1e-8
    union = (outputs | labels).float().sum((1, 2))  # Will be zero if both are 0
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return iou


def evaluate(model, test_dl, device):
    model.segmenter.eval()
    model.counter.eval()

    with torch.no_grad():
        all_iou_scores = []
        all_mse_scores = []
        all_acc_scores = []

        for imgs, masks, counts in test_dl:
            imgs, masks, counts = imgs.to(device), masks.to(device), counts.to(device)
            # Evaluate segmentation
            output_segm = model.segmenter(imgs.unsqueeze(0))

            # Evaluate counts
            output_count = model.counter(torch.cat((imgs.unsqueeze(0), masks.unsqueeze(0).unsqueeze(0)), dim=1)).squeeze(1)

            iou_score = iou(torch.round(output_segm.cpu()), masks.cpu())
            acc_score = int(output_count.item() == counts)
            mse_score = (output_count.item() - counts) ** 2

            all_iou_scores.append(iou_score)
            all_acc_scores.append(acc_score)
            all_mse_scores.append(mse_score)

    print(f'IOU: {sum(all_iou_scores) / len(test_dl) :.2f}, \
          MSE: {sum(all_iou_scores) / len(test_dl) :.2f}, \
          Accuracy: {sum(all_acc_scores) / len(test_dl) :.2f}')
