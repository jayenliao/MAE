import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from torchvision.utils import save_image
from tqdm import tqdm
from einops import rearrange
from model import *
from utils import CSVLogger, setup_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt')
    parser.add_argument("--csv_log", type=str, default="auto",
                        help="Path to CSV file for epoch-level metrics.")
    parser.add_argument("--save_images_dir", type=str, default="logs/cifar10/mae-pretrain/images",
                    help="Directory to save reconstructed image grids.")
    parser.add_argument("--save_images_n", type=int, default=16,
                        help="How many validation images to visualize/save each epoch (must be a square number like 16, 25).")

    args = parser.parse_args()
    if args.save_images_dir:
        if np.sqrt(args.save_images_n) ** 2 != args.save_images_n:
            raise ValueError("save_images_n must be a square number.")
        os.makedirs(args.save_images_dir, exist_ok=True)

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae-pretrain'))
    if args.csv_log == "auto":
        csv_log = os.path.join('logs', 'cifar10', 'mae-pretrain', 'metrics.csv')
        os.makedirs(os.path.dirname(csv_log), exist_ok=True)
    else:
        csv_log = args.csv_log
    csv_logger = CSVLogger(csv_log, fieldnames=["epoch", "train_loss", "lr", "mask_ratio"])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MAE_ViT(mask_ratio=args.mask_ratio).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        for img, label in tqdm(iter(dataloader)):
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('mae_loss', avg_loss, global_step=e)
        csv_logger.log({
            "epoch": e,
            "train_loss": float(avg_loss),
            "lr": float(lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else optim.param_groups[0]["lr"]),
            "mask_ratio": float(args.mask_ratio),
        })
        print(f'In epoch {e}, average training loss is {avg_loss}.')

        '''
        Visualize + save the first N predicted images on val dataset
        '''
        model.eval()
        with torch.no_grad():
            n = args.save_images_n  # take the first n images from val set
            val_img = torch.stack([val_dataset[i][0] for i in range(n)]).to(device)  # [n, C, H, W]
            predicted_val_img, mask = model(val_img)  # both [n, C, H, W]

            # compose the visualization: masked input | reconstructed | original
            # (masked input = visible patches only)
            visible_only = val_img * (1 - mask)
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            visualization = torch.cat([visible_only, predicted_val_img, val_img], dim=0)   # [3n, C, H, W]

            # arrange into a grid: 3 rows (masked/pred/orig) × sqrt(n) columns
            side = int(n ** 0.5)  # assume n is a perfect square (e.g., 16 -> 4)
            img_grid = rearrange(visualization, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', v=3, w1=side)

            # normalize to [0,1] for saving/visualization
            img_grid = (img_grid + 1) / 2
            img_grid = img_grid.clamp(0, 1).detach().cpu()

            # TensorBoard
            writer.add_image('mae_image', img_grid, global_step=e)
            # Save a PNG per epoch
            out_path = os.path.join(args.save_images_dir, f"epoch_{e:04d}.png")
            save_image(img_grid, out_path)   # writes a single big grid image

        '''
        Save model
        '''
        torch.save(model, args.model_path)
