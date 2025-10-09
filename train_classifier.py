import os
import argparse
import math
import time
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model import *
from utils import CSVLogger, setup_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_device_batch_size', type=int, default=256)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--output_root', type=str, default='outputs/')
    parser.add_argument('--exp_name', type=str, default='exp_test')
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--output_model_path', type=str, default='auto')
    parser.add_argument("--csv_log", type=str, default="auto",
                        help="Path to CSV file for epoch-level metrics.")

    args = parser.parse_args()

    # Set up paths
    if args.output_model_path == 'auto':
        if args.pretrained_model_path is not None:
            folder = "pretrain-cls"
            fn = "vit-t-clf-from_pretrained.pt"
        else:
            folder = "scratch-cls"
            fn = "vit-t-clf-from_scratch.pt"
        output_dir = os.path.join(args.output_root, args.exp_name, folder)
        output_model_path = os.path.join(output_dir, fn)
    else:
        output_model_path = args.output_model_path
        output_dir = os.path.dirname(output_model_path)
    os.makedirs(output_dir, exist_ok=True)

    if args.csv_log == 'auto':
        csv_path = os.path.join(output_dir, "metrics.csv")
    else:
        csv_path = args.csv_log
    if args.pretrained_model_path is not None:
        assert os.path.exists(args.pretrained_model_path), f"Pretrained model path {args.pretrained_model_path} does not exist!"
        assert "mae-pretrain" in args.pretrained_model_path, "The pretrained model seems not from mae_pretrain.py"
        print(f"Training classifier using pretrained weights from {args.pretrained_model_path}")
    else:
        print("Training classifier from scratch, without loading any pretrained weights.")
        assert "pretrain-cls" not in output_model_path, "Output model path seems incorrect for training from scratch, please check."

    print(f"Output directory is {output_dir}")
    print(f"Output model will be saved to {output_model_path}")
    print(f"CSV log will be saved to {csv_path}")

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=False, num_workers=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.pretrained_model_path is not None:
        model = torch.load(args.pretrained_model_path, map_location='cpu', weights_only=False)
        writer = SummaryWriter(output_dir)
    else:
        model = MAE_ViT()
        writer = SummaryWriter(output_dir)
    model = ViT_Classifier(model.encoder, num_classes=10).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    csv_logger = CSVLogger(
        csv_path,
        fieldnames=[
            "epoch", "start_time", "end_time", "time_elapsed",
            "train_loss", "train_acc", "val_loss", "val_acc",
            "lr", "from_pretrained"
        ]
    )
    best_val_acc = 0
    step_count = 0
    train_loss_sum = 0.0
    train_correct, train_count = 0, 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        tsStart = time.strftime("%Y%m%d-%H%M%S")
        tStart = time.time()
        model.train()
        losses = []
        acces = []
        for img, label in tqdm(iter(train_dataloader)):
            step_count += 1
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)
            acc = acc_fn(logits, label)
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()

            # accumulate for epoch metrics
            train_loss_sum += loss.item() * img.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == label).sum().item()
            train_count += img.size(0)
            losses.append(loss.item())
            acces.append(acc.item())

        epoch_train_loss = train_loss_sum / train_count
        epoch_train_acc = train_correct / train_count

        lr_scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)
        print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

        model.eval()
        with torch.no_grad():
            losses = []
            acces = []
            for img, label in tqdm(iter(val_dataloader)):
                img = img.to(device)
                label = label.to(device)
                logits = model(img)
                loss = loss_fn(logits, label)
                acc = acc_fn(logits, label)
                losses.append(loss.item())
                acces.append(acc.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)
            print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f'saving best model with acc {best_val_acc} at {e} epoch!')
            torch.save(model, output_model_path)

        tsEnd = time.strftime("%Y%m%d-%H%M%S")
        tEnd = time.time()
        writer.add_scalars('cls/loss', {'train' : avg_train_loss, 'val' : avg_val_loss}, global_step=e)
        writer.add_scalars('cls/acc', {'train' : avg_train_acc, 'val' : avg_val_acc}, global_step=e)
        csv_logger.log({
            "epoch": e,
            "start_time": tsStart,
            "end_time": tsEnd,
            "time_elapsed": tEnd - tStart,
            "train_loss": float(avg_train_loss),
            "train_acc": float(avg_train_acc),
            "val_loss": float(avg_val_loss),
            "val_acc": float(avg_val_acc),
            "lr": float(lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else optim.param_groups[0]["lr"]),
            "from_pretrained": args.pretrained_model_path,
        })
