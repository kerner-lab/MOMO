
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
    train_dataloader: Iterable, val_dataloader: Iterable,
    num_epochs: int, device: torch.device, output_dir: str,
    loss_scaler, optimizer, wandb, args
):

    training_metrics = {
        "epochs_samples": [],
        "train_loss": [], "train_mse": [], "train_ssim": [], "train_lpips": [], "train_gradient": [],
        "val_loss": [], "val_mse": [], "val_ssim": [], "val_lpips": [], "val_gradient": []
    }

    if args.wandb_enabled:
        wandb.define_metric("Training/*", step_metric="train_step")
        wandb.define_metric("Validation/*", step_metric="val_step")
        wandb.define_metric("Training 100k/*", step_metric="global_step")
        wandb.define_metric("Validation 100k/*", step_metric="global_step")

    train_step, val_step, global_step = 0, 0, 0

    with tqdm(range(num_epochs), desc="Epoch") as tqdm_epoch:

        for epoch in tqdm_epoch:

            model.train(True)
            metric_logger = misc.MetricLogger(delimiter="  ")
            metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

            accum_iter = args.accum_iter
            optimizer.zero_grad()

            train_loss, train_mse, train_ssim, train_lpips, train_gradient = 0, 0, 0, 0, 0

            for data_iter_step_train, (samples, _) in enumerate(train_dataloader):

                if data_iter_step_train % accum_iter == 0:
                    lr_sched.adjust_learning_rate(optimizer, data_iter_step_train / len(train_dataloader) + epoch, args)

                samples = samples.to(device, non_blocking=True)

                if args.combined_loss:
                    loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
                    current_loss, loss_dict = loss[0], loss[1]
                    loss_value = current_loss.item()
                    train_loss += loss_value
                    train_mse += loss_dict['mse']
                    train_ssim += loss_dict['ssim']
                    train_lpips += loss_dict['lpips']
                    train_gradient += loss_dict['gradient']

                    if args.wandb_enabled:
                        wandb.log({
                            "train_step": train_step,
                            "Training/Step Loss": loss_value,
                            "Training/Step MSE": loss_dict['mse'],
                            "Training/Step SSIM": loss_dict['ssim'],
                            "Training/Step LPIPS": loss_dict['lpips'],
                            "Training/Step Gradient": loss_dict['gradient']
                        })
                else:
                    loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
                    current_loss = loss[0]
                    loss_value = current_loss.item()
                    train_loss += loss_value

                    if args.wandb_enabled:
                        wandb.log({
                            "train_step": train_step,
                            "Training/Step Loss": loss_value
                        })
                train_step += 1

                if not math.isfinite(loss_value):
                    print("Training Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

                current_loss /= accum_iter
                loss_scaler(current_loss, optimizer, parameters=model.parameters(),
                            update_grad=(data_iter_step_train + 1) % accum_iter == 0)
                if (data_iter_step_train + 1) % accum_iter == 0:
                    optimizer.zero_grad()

                torch.cuda.synchronize()

                lr = optimizer.param_groups[0]["lr"]
                metric_logger.update(lr=lr)

                samples_seen_train = (data_iter_step_train + 1) * args.batch_size
                if samples_seen_train % args.evaluation_interval < args.batch_size:

                    model.eval()
                    val_loss, val_mse, val_ssim, val_lpips, val_gradient = 0, 0, 0, 0, 0

                    with torch.no_grad():
                        for data_iter_step_val, (samples, _) in enumerate(val_dataloader):

                            samples = samples.to(device, non_blocking=True)

                            if args.combined_loss:
                                loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
                                current_loss, loss_dict = loss[0], loss[1]
                                loss_value = current_loss.item()
                                val_loss += loss_value
                                val_mse += loss_dict['mse']
                                val_ssim += loss_dict['ssim']
                                val_lpips += loss_dict['lpips']
                                val_gradient += loss_dict['gradient']

                                if args.wandb_enabled:
                                    wandb.log({
                                        "val_step": val_step,
                                        "Validation/Step Loss": loss_value,
                                        "Validation/Step MSE": loss_dict['mse'],
                                        "Validation/Step SSIM": loss_dict['ssim'],
                                        "Validation/Step LPIPS": loss_dict['lpips'],
                                        "Validation/Step Gradient": loss_dict['gradient']
                                    })
                            else:
                                loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
                                current_loss = loss[0]
                                loss_value = current_loss.item()
                                val_loss += loss_value

                                if args.wandb_enabled:
                                    wandb.log({
                                        "val_step": val_step,
                                        "Validation/Step Loss": loss_value
                                    })
                            val_step += 1

                            if not math.isfinite(loss_value):
                                print("Validation Loss is {}, stopping training".format(loss_value))
                                sys.exit(1)

                            samples_seen_val = (data_iter_step_val + 1) * args.batch_size
                            if samples_seen_val % args.evaluation_interval < args.batch_size:
                                break

                    avg_train_loss = train_loss / (data_iter_step_train + 1)
                    avg_val_loss = val_loss / (data_iter_step_val + 1)

                    training_metrics["epochs_samples"].append(f"{epoch}-{samples_seen_train}")
                    training_metrics["train_loss"].append(float(avg_train_loss))
                    training_metrics["val_loss"].append(float(avg_val_loss))

                    if args.combined_loss:
                        avg_train_mse = train_mse / (data_iter_step_train + 1)
                        avg_train_ssim = train_ssim / (data_iter_step_train + 1)
                        avg_train_lpips = train_lpips / (data_iter_step_train + 1)
                        avg_train_gradient = train_gradient / (data_iter_step_train + 1)
                        avg_val_mse = val_mse / (data_iter_step_val + 1)
                        avg_val_ssim = val_ssim / (data_iter_step_val + 1)
                        avg_val_lpips = val_lpips / (data_iter_step_val + 1)
                        avg_val_gradient = val_gradient / (data_iter_step_val + 1)

                        training_metrics["train_mse"].append(float(avg_train_mse))
                        training_metrics["train_ssim"].append(float(avg_train_ssim))
                        training_metrics["train_lpips"].append(float(avg_train_lpips))
                        training_metrics["train_gradient"].append(float(avg_train_gradient))
                        training_metrics["val_mse"].append(float(avg_val_mse))
                        training_metrics["val_ssim"].append(float(avg_val_ssim))
                        training_metrics["val_lpips"].append(float(avg_val_lpips))
                        training_metrics["val_gradient"].append(float(avg_val_gradient))

                    metrics_file = os.path.join(output_dir, f"{args.name_of_run}-training_metrics.json")
                    with open(metrics_file, 'w') as f:
                        json.dump(training_metrics, f, indent=4)

                    if args.wandb_enabled:
                        log_dict = {
                            "Training 100k/Loss": avg_train_loss,
                            "Validation 100k/Loss": avg_val_loss,
                            "Epoch": epoch,
                            "global_step": global_step
                        }
                        if args.combined_loss:
                            log_dict.update({
                                "Training 100k/MSE": avg_train_mse,
                                "Training 100k/SSIM": avg_train_ssim,
                                "Training 100k/LPIPS": avg_train_lpips,
                                "Training 100k/Gradient": avg_train_gradient,
                                "Validation 100k/MSE": avg_val_mse,
                                "Validation 100k/SSIM": avg_val_ssim,
                                "Validation 100k/LPIPS": avg_val_lpips,
                                "Validation 100k/Gradient": avg_val_gradient
                            })
                        wandb.log(log_dict)
                        global_step += 1

                    if args.combined_loss:
                        print(f"Epoch [{epoch+1}/{num_epochs}]")
                        print(f"  Train Loss: {avg_train_loss:.4f} | Train - MSE: {avg_train_mse:.4f}, SSIM: {avg_train_ssim:.4f}, LPIPS: {avg_train_lpips:.4f}, Gradient: {avg_train_gradient:.4f}")
                        print(f"  Val Loss: {avg_val_loss:.4f} | Val - MSE: {avg_val_mse:.4f}, SSIM: {avg_val_ssim:.4f}, LPIPS: {avg_val_lpips:.4f}, Gradient: {avg_val_gradient:.4f}\n")
                    else:
                        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

                    model.train(True)
                    optimizer.zero_grad()
                    train_loss, train_mse, train_ssim, train_lpips, train_gradient = 0, 0, 0, 0, 0

                    misc.save_model(args=args, output_dir=output_dir, save_name=f"checkpoint_{args.name_of_run}-{epoch}-{samples_seen_train}", model=model)
                    misc.save_model(args=args, output_dir=output_dir, save_name=f"checkpoint_{args.name_of_run}-last", model=model)

    print(f"Training metrics saved to {metrics_file}")
