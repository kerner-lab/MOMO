
import os
import json
import math
import sys
from tqdm import tqdm

import torch
from typing import Iterable

import utils.misc as misc
import utils.lr_sched as lr_sched


def model_training(
    model: torch.nn.Module,
    train_dataloader: Iterable, val_dataloader:Iterable,
    num_epochs: int, device: torch.device, output_dir: str,
    criterion, optimizer, wandb, args
):

    # Dictionary to store metrics
    training_metrics = {
        "epochs": [],
        "train_loss": [],
        "val_loss": []
    }

    with tqdm(range(num_epochs), desc="Epoch") as tqdm_epoch:

        for epoch in tqdm_epoch:

            # Training
            model.train()
            train_loss = 0
            for images, _ in train_dataloader:
                images = images.to(device)
                reconstructions = model(images)
                loss = criterion(reconstructions, images)
                train_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss /= len(train_dataloader)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, _ in val_dataloader:
                    images = images.to(device)
                    reconstructions = model(images)
                    val_loss += criterion(reconstructions, images).item()
            val_loss /= len(val_dataloader)

            # Store metrics
            training_metrics["epochs"].append(epoch)
            training_metrics["train_loss"].append(float(train_loss.item()))
            training_metrics["val_loss"].append(float(val_loss))

            # Log to wandb
            if args.wandb_enabled:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                })
            print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss:.4f}")

            # Save the encoder part of the model
            if ((epoch) % 1 == 0) or (epoch == 0):
                # Save the encoder and decoder part of the model separately
                torch.save(model.encoder.state_dict(), os.path.join(output_dir, f"{args.name_of_run}-{epoch}.pth"))
                torch.save(model.decoder.state_dict(), os.path.join(output_dir, f"{args.name_of_run}-decoder-{epoch}.pth"))


    # Save training metrics to JSON file
    metrics_file = os.path.join(output_dir, f"{args.name_of_run}-training_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(training_metrics, f, indent=4)

    print(f"Training metrics saved to {metrics_file}")


def model_training_vit(
    model: torch.nn.Module,
    model_without_ddp: torch.nn.Module,
    train_dataloader: Iterable, val_dataloader: Iterable,
    num_epochs: int, device: torch.device, output_dir: str,
    loss_scaler, optimizer, wandb, args
):

    # Dictionary to store metrics
    training_metrics = {
        "epochs": [],
        "train_loss": [],
        "val_loss": []
    }

    with tqdm(range(num_epochs), desc="Epoch") as tqdm_epoch:

        for epoch in tqdm_epoch:

            # Training
            model.train(True)
            metric_logger = misc.MetricLogger(delimiter="  ")
            metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            header = 'Epoch: [{}]'.format(epoch)
            print_freq = 20

            accum_iter = args.accum_iter
            optimizer.zero_grad()

            for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)):

                # we use a per iteration (instead of per epoch) lr scheduler
                if data_iter_step % accum_iter == 0:
                    lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(train_dataloader) + epoch, args)

                samples = samples.to(device, non_blocking=True)

                loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
                loss_value = loss.item()

                if not math.isfinite(loss_value):
                    print("Training Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

                loss /= accum_iter
                loss_scaler(loss, optimizer, parameters=model.parameters(),
                            update_grad=(data_iter_step + 1) % accum_iter == 0)
                if (data_iter_step + 1) % accum_iter == 0:
                    optimizer.zero_grad()

                torch.cuda.synchronize()

                metric_logger.update(loss=loss_value)

                lr = optimizer.param_groups[0]["lr"]
                metric_logger.update(lr=lr)

            current_training_loss = metric_logger.meters["loss"].global_avg

            # Validation
            model.eval()
            with torch.no_grad():
                metric_logger = misc.MetricLogger(delimiter="  ")
                header = 'Validation Epoch: [{}]'.format(epoch)
                print_freq = 20

                accum_iter = args.accum_iter

                for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(val_dataloader, print_freq, header)):

                    samples = samples.to(device, non_blocking=True)

                    loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
                    loss_value = loss.item()

                    if not math.isfinite(loss_value):
                        print("Validation Loss is {}, stopping training".format(loss_value))
                        sys.exit(1)

                    metric_logger.update(loss=loss_value)

            current_validation_loss = metric_logger.meters["loss"].global_avg

            # Store metrics
            training_metrics["epochs"].append(epoch)
            training_metrics["train_loss"].append(float(current_training_loss))
            training_metrics["val_loss"].append(float(current_validation_loss))

            # Log to wandb
            if args.wandb_enabled:
                wandb.log(
                    {
                        "Training Loss": current_training_loss,
                        "Validation Loss": current_validation_loss,
                        "Epoch": epoch
                    }
                )
            print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {current_training_loss:.4f}, Val Loss: {current_validation_loss:.4f}")

            # Save the encoder part of the model
            if ((epoch) % 1 == 0) or (epoch == 0):
                misc.save_model(args=args, output_dir=output_dir, save_name=f"{args.name_of_run}-{epoch}",model=model)

    # Save training metrics to JSON file
    metrics_file = os.path.join(output_dir, f"{args.name_of_run}-training_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(training_metrics, f, indent=4)

    print(f"Training metrics saved to {metrics_file}")
