import argparse, os, math, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader

from model import MAE_ViT, ViT_Classifier

CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def to01(x):  # [-1,1] -> [0,1]
    return ((x.clamp(-1,1) + 1) / 2).clamp(0,1)

@torch.no_grad()
def predict(model, x, device):
    logits = model(x.to(device))
    preds = logits.argmax(dim=1).cpu()
    return preds

def load_classifier(ckpt_path, device='cuda'):
    if ckpt_path and os.path.isfile(ckpt_path):
        model = torch.load(ckpt_path, map_location=device, weights_only=False)
        print(f'Loaded: {ckpt_path}')
        model.eval()
        return model
    print('WARNING: checkpoint missing or None. Using randomly initialized weights')

def pick_indices(ds_len, n, indices, use_first_n, seed):
    if indices:
        return indices
    if use_first_n:
        return list(range(min(n, ds_len)))
    rng = np.random.RandomState(seed)
    return rng.choice(ds_len, size=min(n, ds_len), replace=False).tolist()

def plot_figure(images, labels, preds, title, save_path, ncols=6):
    n = images.size(0)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1*ncols, 3.6*nrows), constrained_layout=False)
    axes = axes.flatten() if nrows*ncols > 1 else [axes]

    # vertical positions (in Axes coordinates) for the two text lines
    Y_TRUE = 1.16   # top line
    Y_PRED = 1.04   # second line, directly below the true label

    for k, ax in enumerate(axes):
        if k >= n:
            ax.axis('off')
            continue

        img01 = to01(images[k].cpu()).permute(1,2,0).numpy()
        y, p = int(labels[k]), int(preds[k])
        ok = (p == y)

        ax.imshow(img01)
        ax.axis('off')

        # draw BOTH lines as text above the image, with fixed positions
        ax.text(0.5, Y_TRUE, f'True: {CLASSES[y]}',
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=14, color='black', # fontweight='bold',
                clip_on=False)

        ax.text(0.5, Y_PRED, f'Pred: {CLASSES[p]}',
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=14, color=('green' if ok else 'red'),
                clip_on=False)

    # leave headroom for the two lines & the suptitle
    fig.suptitle(title, fontsize=20, y=0.995, fontweight='bold')
    fig.subplots_adjust(top=0.84, hspace=0.45, wspace=0.2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=220, bbox_inches='tight')
    print(f'Saved: {save_path}')

def make_loaders(batch_size=128):
    val_dataset = CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return val_dataset, val_dataloader

def gather_examples(model_pre, model_scr, dataloader, device, per_category=6, max_batches=None):
    """
    Build four buckets of (img, label, pred_pre, pred_scr):
      A: pretrained correct, scratch wrong
      B: scratch correct, pretrained wrong
      C: both correct
      D: both wrong
    """
    buckets = {'pre_wins': [], 'scr_wins': [], 'both_correct': [], 'both_wrong': []}
    need_more = lambda: any(len(v) < per_category for v in buckets.values())

    for b, (x, y) in enumerate(dataloader):
        if not need_more():
            break
        pre = predict(model_pre, x, device)
        scr = predict(model_scr, x, device)

        for i in range(x.size(0)):
            xi, yi = x[i], y[i].item()
            pre_ok = (pre[i].item() == yi)
            scr_ok = (scr[i].item() == yi)
            item = (xi, yi, pre[i].item(), scr[i].item())

            if pre_ok and not scr_ok and len(buckets['pre_wins']) < per_category:
                buckets['pre_wins'].append(item)
            elif scr_ok and not pre_ok and len(buckets['scr_wins']) < per_category:
                buckets['scr_wins'].append(item)
            elif pre_ok and scr_ok and len(buckets['both_correct']) < per_category:
                buckets['both_correct'].append(item)
            elif (not pre_ok) and (not scr_ok) and len(buckets['both_wrong']) < per_category:
                buckets['both_wrong'].append(item)

        if max_batches is not None and (b+1) >= max_batches:
            break
    return buckets

def _tile(ax, img01, title_top, title_bottom, color_top='black', color_bottom='black'):
    ax.imshow(img01.permute(1,2,0).numpy())
    ax.axis('off')
    ax.set_title(title_top, color=color_top, fontsize=10, pad=2)
    ax.text(0.5, -0.15, title_bottom, transform=ax.transAxes,
            ha='center', va='top', fontsize=9, color=color_bottom)

def plot_bucket(bucket, title, savepath=None, ncols=6):
    n = len(bucket)
    if n == 0:
        print(f'[WARN!!] No samples for {title}')
        return
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes = axes.flatten() if nrows*ncols>1 else [axes]

    for k, ax in enumerate(axes):
        if k >= n:
            ax.axis('off')
            continue
        x, y, pre, scr = bucket[k]
        img01 = to01(x.cpu())
        # captions
        top = f'True: {CLASSES[y]}'
        # color green if correct, red if wrong (pretrained line on top, scratch below)
        pre_ok = (pre == y)
        scr_ok = (scr == y)
        bottom = f'Pretrain: {CLASSES[pre]}   |   Scratch: {CLASSES[scr]}'
        _tile(ax, img01, title_top=top,
              title_bottom=bottom,
              color_top='green' if (pre_ok or scr_ok) else 'red',
              color_bottom='green' if pre_ok and scr_ok else ('orange' if pre_ok or scr_ok else 'red'))
    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=200)
        print(f'Saved: {savepath}')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pretrained_ckpt', type=str, required=True,
                        help='Checkpoint of classifier fine-tuned with MAE pretraining')
    parser.add_argument('--scratch_ckpt', type=str, required=True,
                        help='Checkpoint of classifier trained from scratch')
    parser.add_argument('--n', type=int, default=12, help='number of images to visualize')
    parser.add_argument('--indices', type=int, nargs='+', default=None, help='overrides first-n & seed')
    parser.add_argument('--use_first_n', action='store_true', help='use the first n images (deterministic). If false and no indices, sample by seed.')
    parser.add_argument('--seed', type=int, default=0, help='seed for deterministic sampling when not using first-n')
    parser.add_argument('--save_dir', type=str, default='logs/cls_viz', help='directory to save the plot images')
    args = parser.parse_args()

    device = args.device
    val_dataset = CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))

    # pick the same indices once
    idx = pick_indices(len(val_dataset), args.n, args.indices, args.use_first_n, args.seed)
    xs, ys = zip(*[val_dataset[i] for i in idx])
    images = torch.stack(xs, 0) # [n, 3, H, W]
    labels = torch.tensor(ys)   # [n]

    # load models
    pre = load_classifier(args.pretrained_ckpt, device)
    scr = load_classifier(args.scratch_ckpt, device)

    # predict on the same images
    with torch.no_grad():
        preds_pre = predict(pre, images, device)
        preds_scr = predict(scr, images, device)

    # make two separate figures
    os.makedirs(args.save_dir, exist_ok=True)
    plot_figure(images, labels, preds_pre,
                title='Predictions — Pretrained Classifier',
                save_path=os.path.join(args.save_dir, 'pretrained.png'))
    plot_figure(images, labels, preds_scr,
                title='Predictions — Scratch Classifier',
                save_path=os.path.join(args.save_dir, 'scratch.png'))

    # (optional) also dump the chosen indices so you can reuse exactly the same set later
    with open(os.path.join(args.save_dir, 'indices.txt'), 'w') as f:
        f.write(','.join(map(str, idx)))

if __name__ == '__main__':
    main()
