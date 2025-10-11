import os
import time
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
from model import *
from utils import CSVLogger, setup_seed

def to01(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1) / 2).clamp(0, 1)

def ensure_mask_channels(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 4 and mask.size(1) == 1 and x.size(1) != 1:
        mask = mask.expand(-1, x.size(1), -1, -1)
    return mask

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
    parser.add_argument('--output_root', type=str, default='outputs/')
    parser.add_argument('--exp_name', type=str, default='exp_test')
    parser.add_argument('--model_fn', type=str, default='vit-t-mae.pt')
    parser.add_argument("--csv_fn", type=str, default="metrics.csv",
                        help="CSV file name for epoch-level metrics.")

    # Visualization
    parser.add_argument("--save_images_dir", type=str, default="images/",
                    help="Directory to save reconstructed image grids.")
    parser.add_argument("--save_images_n", type=int, default=24,
                        help="How many validation images to visualize/save each epoch.")
    parser.add_argument("--visualize_freq", type=int, default=10)  # 0 = off. also runs on last epoch
    parser.add_argument("--pad", type=int, default=3)             # gutter (pixels) between the 3 tiles
    parser.add_argument("--pad_value", type=float, default=1)

    # Ablations
    parser.add_argument('--decoder_layers', type=int, default=4)
    parser.add_argument('--decoder_dim', type=int, default=192, help='decoder width; paper often uses 512')
    parser.add_argument('--enc_mask_token', action='store_true',
                        help='use mask tokens in the encoder (Table 1c ON)')
    parser.add_argument('--mask_strategy', type=str, default='random',
                        choices=['random','block','grid'], help='Table 1f')

    args = parser.parse_args()

    # Set up paths
    output_dir      = os.path.join(args.output_root, args.exp_name, "mae-pretrain")
    model_path      = os.path.join(output_dir, args.model_fn)
    csv_path        = os.path.join(output_dir, args.csv_fn)
    save_images_dir = os.path.join(output_dir, args.save_images_dir)
    writer_dir      = os.path.join(output_dir, 'tensorboard')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_images_dir, exist_ok=True)
    os.makedirs(writer_dir, exist_ok=True)

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    writer = SummaryWriter(writer_dir)
    csv_logger = CSVLogger(
        csv_path,
        fieldnames=[
            "epoch", "time_elapsed", "end_time", "start_time",
            "train_loss", "lr",
            "decoder_layers", "decoder_dim",
            "mask_strategy", "mask_ratio", "enc_mask_token",
        ],
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MAE_ViT(
        mask_ratio=args.mask_ratio,
        decoder_layers=args.decoder_layers,
        decoder_dim=args.decoder_dim,
        enc_mask_token=args.enc_mask_token,
        mask_strategy=args.mask_strategy
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    step_count = 0
    optim.zero_grad()
    for e in range(1, args.total_epoch + 1):
        tsStart = time.strftime("%Y%m%d-%H%M%S")
        tStart = time.time()
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
        tsEnd = time.strftime("%Y%m%d-%H%M%S")
        tEnd = time.time()
        writer.add_scalar('mae_loss', avg_loss, global_step=e)
        csv_logger.log({
            "epoch": e,
            "start_time": tsStart,
            "end_time": tsEnd,
            "time_elapsed": tEnd - tStart,
            "train_loss": float(avg_loss),
            "lr": float(lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else optim.param_groups[0]["lr"]),
            "decoder_layers": args.decoder_layers,
            "decoder_dim": args.decoder_dim,
            "mask_strategy": args.mask_strategy,
            "mask_ratio": float(args.mask_ratio),
            "enc_mask_token": bool(args.enc_mask_token),
        })
        print(f'In epoch {e}, average training loss is {avg_loss}.')

        '''
        Visualize + save the first N predicted images on val dataset
        '''
        visualize = (args.visualize_freq > 0 and (e % args.visualize_freq == 0)) or (e == args.total_epoch - 1)
        if visualize:
            save_images_dir_epoch = os.path.join(save_images_dir, f"epoch_{e:04d}")
            os.makedirs(save_images_dir_epoch, exist_ok=True)
            model.eval()
            with torch.inference_mode():
                n = args.save_images_n
                val_imgs = torch.stack([val_dataset[i][0] for i in range(n)]).to(device)  # [n,C,H,W]
                preds, masks = model(val_imgs)  # [n,C,H,W], [n,1/H,W] depending on impl
                masks = ensure_mask_channels(masks, val_imgs)

                masked = val_imgs * (1 - masks)
                masked = masked + masks * 0
                recon  = preds * masks + val_imgs * (1 - masks)

                # write n separate PNGs: each is [masked | recon | original] with spacing
                for i in range(n):
                    tiles01 = torch.stack([
                        to01(masked[i]),     # masked   (left)
                        to01(recon[i]),      # recon    (middle)
                        to01(val_imgs[i]),   # original (right)
                    ], dim=0)  # [3,C,H,W]

                    grid = torchvision.utils.make_grid(
                        tiles01, nrow=3, padding=args.pad, pad_value=args.pad_value
                    )  # [C, H, 3W + gutters]

                    out_path = os.path.join(save_images_dir_epoch, f"epoch_{e:04d}_idx_{i:03d}.png")
                    torchvision.utils.save_image(grid, out_path)

                    # writer.add_image(f"mae_single/{i:03d}", grid, global_step=e)
        '''
        Save model
        '''
        torch.save(model, model_path)
