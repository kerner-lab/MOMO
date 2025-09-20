
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
from torch import inf


def save_model(args, output_dir, save_name, model):
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / (f"{save_name}.pth")
    to_save = {
        'model': model.state_dict(),
        'args': args,
    }
    torch.save(to_save, checkpoint_path)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.if_pretrained:
        if args.vit_pretrained_checkpoint_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.vit_pretrained_checkpoint_path, map_location='cpu', check_hash=True)
        else:
            if "vit-b" in args.train_model:
                checkpoint = torch.load(os.path.join(args.vit_pretrained_checkpoint_path, "mae_pretrain_vit_base.pth"), map_location='cpu')
            else:
                checkpoint = torch.load(os.path.join(args.vit_pretrained_checkpoint_path, "mae_pretrain_vit_large.pth"), map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Resume checkpoint %s" % args.vit_pretrained_checkpoint_path)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
