
import cv2
import os
import math
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score,
    precision_score, recall_score
)
import sys
from tqdm import tqdm
from typing import Iterable
import wandb

import segmentation_models_pytorch as smp
import torch
from torch.autograd import Variable

import utils.lr_sched as lr_sched
import utils.misc as misc


''' Classification '''

def training_model_classification(model: torch.nn.Module,
                train_dataloader: Iterable, val_dataloader: Iterable,
                optimizer: torch.optim.Optimizer, device: torch.device,
                output_dir: str, patience: int,
                name_of_run: str, criterion, args):

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(args.num_epochs), desc=name_of_run):

        # Training loop
        model.train()
        train_loss = 0.0

        for inputs, labels, _ in tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{args.num_epochs}] Training"):
            inputs = Variable(inputs.type(torch.FloatTensor)).to(device)
            labels = Variable(labels.type(torch.LongTensor)).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # Validation loop
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels, _ in tqdm(val_dataloader, desc=f"Epoch [{epoch+1}/{args.num_epochs}] Validation"):
                inputs = Variable(inputs.type(torch.FloatTensor)).to(device)
                labels = Variable(labels.type(torch.LongTensor)).to(device)

                outputs = model(inputs)
                labels = labels.squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Handle both binary and multi-class cases
                if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.shape[1] == 1):
                    # Binary classification
                    predicted = (outputs > 0.5).long()
                else:
                    # Multi-class classification
                    _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels.long()).sum().item()

        val_loss /= len(val_dataloader)
        val_accuracy = 100 * correct / total

        # Print logs and update wandb
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        if args.wandb_enabled:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, f"checkpoint-best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")

    # Save the last checkpoint
    torch.save(model.state_dict(), os.path.join(output_dir, f"checkpoint-last.pth"))

    return model


def training_model_classification_vit(model: torch.nn.Module,
                train_dataloader: Iterable, val_dataloader: Iterable,
                optimizer: torch.optim.Optimizer, device: torch.device,
                output_dir: str, patience: int, name_of_run: str,
                criterion, loss_scaler, args):

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(args.num_epochs), desc=name_of_run):

        # Training loop
        model.train(True)
        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 20

        accum_iter = args.accum_iter
        optimizer.zero_grad()

        for data_iter_step, (samples, targets, _) in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)):

            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(train_dataloader) + epoch, args)

            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= accum_iter
            loss_scaler(loss, optimizer, clip_grad=args.max_norm,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            current_training_loss = metric_logger.meters["loss"].global_avg

        # Validation loop
        model.eval()
        with torch.no_grad():
            metric_logger = misc.MetricLogger(delimiter="  ")
            header = 'Validation Epoch: [{}]'.format(epoch)
            print_freq = 20

            accum_iter = args.accum_iter

            ground_truth, prediction = [] ,[]
            for data_iter_step, (samples, targets, _) in enumerate(metric_logger.log_every(val_dataloader, print_freq, header)):
                samples = samples.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = model(samples)
                    loss = criterion(outputs, targets)

                loss_value = loss.item()

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

                metric_logger.update(loss=loss_value)

                output = torch.nn.functional.softmax(outputs, dim=1)
                label = output.cpu().detach().numpy()
                prediction.append(np.argmax(label))
                ground_truth.append(targets.cpu().numpy()[0])

                current_validation_loss = metric_logger.meters["loss"].global_avg

        # Print logs and update wandb
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {current_training_loss:.4f}, Val Loss: {current_validation_loss:.4f}, Val Accuracy: {accuracy_score(ground_truth, prediction):.4f}")
        if args.wandb_enabled:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": current_training_loss,
                "val_loss": current_validation_loss,
                "val_accuracy": accuracy_score(ground_truth, prediction)
            })

        # Early stopping check
        if current_validation_loss < best_val_loss:
            best_val_loss = current_validation_loss
            patience_counter = 0
            misc.save_model(
                args=args, output_dir=output_dir, model=model, model_without_ddp=model, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch="best")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")

    # Save the last checkpoint
    misc.save_model(
        args=args, output_dir=output_dir, model=model, model_without_ddp=model, optimizer=optimizer,
        loss_scaler=loss_scaler, epoch="last")

    return model


@torch.no_grad()
def evaluate_model_classification(
    model: torch.nn.Module, test_dataloader: Iterable,
    device: torch.device, result_csv_path: str, balance_data: str,
    config: dict, pretraining_configuration: str, no_of_samples: int,
    args):

    model.to(device)
    model.eval()

    with torch.no_grad():

        ground_truth, prediction = [], []
        for data_iter_step, (samples, targets, _) in enumerate(test_dataloader):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(samples)

            output = torch.nn.functional.softmax(outputs, dim=1)
            label = output.cpu().detach().numpy()
            prediction.append(np.argmax(label))
            ground_truth.append(targets.cpu().numpy()[0])

    print("-"*60)
    print("Accuracy:", accuracy_score(ground_truth, prediction))
    print("Precision:", precision_score(ground_truth, prediction, average="weighted"))
    print("Recall:", recall_score(ground_truth, prediction, average="weighted"))
    print("F1-Score:", f1_score(ground_truth, prediction, average="weighted"))

    ### Plot classification report
    label_dict = config["label_dict"]
    label_dict_reverse = config["label_dict_reverse"]
    label_dict_reverse = {int(k): v for k, v in label_dict_reverse.items()}

    ground_truth = [*map(label_dict_reverse.get, ground_truth)]
    prediction = [*map(label_dict_reverse.get, prediction)]

    print(classification_report(ground_truth, prediction))
    print("-"*60)

    ### Plot confusion matrix
    cmtx = pd.DataFrame(
        confusion_matrix(ground_truth, prediction, labels=list(label_dict.keys())), 
        index=list(label_dict.keys()),
        columns=list(label_dict.keys())
    )
    print(cmtx)
    print("-"*60)

    if os.path.exists(result_csv_path):
        result_df = pd.read_csv(result_csv_path)
    else:
        result_df = pd.DataFrame(columns=[
            "Downstream Task", "Training type",
            "Train Model", "Pre-training configuration",
            "balance_data", "no_of_training_samples",
            "Accuracy", "Precision",
            "Recall", "F1-Score"
        ])
    current_result = [
        args.dataset, args.which_pretraining,
        args.train_model, pretraining_configuration, balance_data, no_of_samples,
        round(accuracy_score(ground_truth, prediction), 4), round(precision_score(ground_truth, prediction, average="weighted"), 4),
        round(recall_score(ground_truth, prediction, average="weighted"), 4), round(f1_score(ground_truth, prediction, average="weighted"), 4)
    ]
    result_df.loc[len(result_df)] = current_result
    result_df.to_csv(result_csv_path, index=False)

    return accuracy_score(ground_truth, prediction), precision_score(ground_truth, prediction, average="weighted"), recall_score(ground_truth, prediction, average="weighted"), f1_score(ground_truth, prediction, average="weighted")


''' Segmentation '''

def training_model_segmentation(model: torch.nn.Module,
                train_dataloader: Iterable, val_dataloader: Iterable,
                optimizer: torch.optim.Optimizer, device: torch.device,
                num_classes: int, output_dir: str, patience: int,
                name_of_run: str, criterion, args):

    for epoch in tqdm(range(args.num_epochs), desc=name_of_run):

        # Training loop
        model.train()
        train_loss = 0.0
        for inputs, labels, _ in tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{args.num_epochs}] Training"):
            inputs = Variable(inputs.type(torch.FloatTensor)).to(device)
            if num_classes == 1:
                labels = Variable(labels.type(torch.FloatTensor)).to(device)
            else:
                labels = Variable(labels.type(torch.LongTensor)).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # Validation loop
        model.eval()
        val_loss, val_iou = 0.0, 0.0

        with torch.no_grad():
            for inputs, labels, _ in tqdm(val_dataloader, desc=f"Epoch [{epoch+1}/{args.num_epochs}] Validation"):
                inputs = Variable(inputs.type(torch.FloatTensor)).to(device)
                if num_classes == 1:
                    labels = Variable(labels.type(torch.FloatTensor)).to(device)
                else:
                    labels = Variable(labels.type(torch.LongTensor)).to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                if num_classes == 1:
                    posterior = torch.sigmoid(outputs)
                    prediction = torch.where(posterior > 0.5, 1, 0)
                else:
                    posterior = torch.softmax(outputs, dim=1)
                    prediction = torch.argmax(posterior)
                tp, fp, fn, tn = smp.metrics.get_stats(prediction, labels.type(torch.int64), mode='binary')
                val_iou += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise").item()

        val_loss /= len(val_dataloader)
        val_iou /= len(val_dataloader)

        # Print logs and update wandb
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        if args.wandb_enabled:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_iou
            })

        # Save finetuned model
        if ((epoch + 1) % 2 == 0) or (epoch == 0):
            torch.save(model.state_dict(), os.path.join(output_dir, f"checkpoint_{epoch+1}.pth"))

    return model


@torch.no_grad()
def evaluate_model_segmentation(
    model: torch.nn.Module, test_dataloader: Iterable,
    device: torch.device, output_dir: str,
    result_csv_path: str, config: dict,
    pretraining_configuration: str, args):

    model.to(device)
    model.eval()

    with torch.no_grad():

        os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
        pixel_iou, pixel_accuracy, pixel_precision, pixel_recall, pixel_dice = [], [], [], [], []

        for _, (inputs, labels, filename) in enumerate(tqdm(test_dataloader)):

            inputs = Variable(inputs.type(torch.FloatTensor)).to(device)
            if config["num_classes"] == 1:
                labels = Variable(labels.type(torch.FloatTensor)).to(device)
            else:
                labels = Variable(labels.type(torch.LongTensor)).to(device)

            outputs = model(inputs)

            if config["num_classes"] == 1:
                posterior = torch.sigmoid(outputs)
                prediction = torch.where(posterior > 0.5, 1, 0)
            else:
                posterior = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(posterior)

            cv2.imwrite(os.path.join(output_dir, "predictions", filename[0]), prediction.cpu().numpy()[0].squeeze().astype(np.uint8))

            tp, fp, fn, tn = smp.metrics.get_stats(prediction, labels.type(torch.int64), mode='binary')

            pixel_iou.append(smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise").item())
            pixel_accuracy.append(smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro-imagewise").item())
            pixel_recall.append(smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise").item())
            pixel_precision.append(smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise").item())
            current_dice_coff = (2 * tp) / ((2* tp) + fp + fn)
            pixel_dice.append(current_dice_coff.item())


    print("-"*60)
    print("Pixel IoU:", sum(pixel_iou) / len(pixel_iou))
    print("Pixel Accuracy:", sum(pixel_accuracy) / len(pixel_accuracy))
    print("Pixel Precision:", sum(pixel_recall) / len(pixel_recall))
    print("Pixel Recall:", sum(pixel_precision) / len(pixel_precision))
    print("Pixel Dice", sum(pixel_dice) / len(pixel_dice))

    result_csv_path = os.path.join("results", f"results_segmentation.csv")
    if os.path.exists(result_csv_path):
        result_df = pd.read_csv(result_csv_path)
    else:
        result_df = pd.DataFrame(columns=[
            "Downstream Task", "Training type",
            "Train Model", "Pre-training configuration",
            "Pixel IoU", "Pixel Accuracy",
            "Pixel Precision", "Pixel Recall",
            "Pixel Dice"
        ])
    current_result = [
        args.dataset, args.which_pretraining,
        args.train_model, pretraining_configuration,
        sum(pixel_iou) / len(pixel_iou), sum(pixel_accuracy) / len(pixel_accuracy),
        sum(pixel_precision) / len(pixel_precision), sum(pixel_recall) / len(pixel_recall),
        sum(pixel_dice) / len(pixel_dice)
    ]
    result_df.loc[len(result_df)] = current_result
    result_df.to_csv(result_csv_path, index=False)

    return sum(pixel_iou) / len(pixel_iou), sum(pixel_accuracy) / len(pixel_accuracy), sum(pixel_recall) / len(pixel_recall), sum(pixel_precision) / len(pixel_precision), sum(pixel_dice) / len(pixel_dice)
