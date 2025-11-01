
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
from timm.utils import accuracy
from tqdm import tqdm
from typing import Iterable
import wandb

import segmentation_models_pytorch as smp
import torch

import utils.lr_sched as lr_sched
from utils.metrics import compute_object_metrics
import utils.misc as misc


''' Classification '''

def save_results(metrics_dir, name_of_run, eval_accuracy, eval_precision, eval_recall, eval_f1score, eval_acc1, eval_acc5, cls_report, cmtx):

    file1 = open(os.path.join(metrics_dir, "results.txt"), "w")
    file1.write(name_of_run)
    file1.write("\n")
    file1.write("-"*60)
    file1.write("\n")
    file1.write("Accuracy:  ")
    file1.write(str(eval_accuracy))
    file1.write("\n")
    file1.write("Precision:  ")
    file1.write(str(eval_precision))
    file1.write("\n")
    file1.write("Recall:  ")
    file1.write(str(eval_recall))
    file1.write("\n")
    file1.write("F1-Score:  ")
    file1.write(str(eval_f1score))
    file1.write("\n")
    file1.write("Top-1 Accuracy:  ")
    file1.write(str(eval_acc1))
    file1.write("\n")
    file1.write("Top-5 Accuracy:  ")
    file1.write(str(eval_acc5))
    file1.write("\n")
    file1.write("-"*60)
    file1.write("\n")
    file1.write(cls_report)
    file1.write("\n")
    file1.write("-"*60)
    file1.write("\n")
    file1.write(cmtx.to_string())
    file1.write("\n")
    file1.write("-"*60)
    file1.write("\n")
    file1.close()


def training_model_classification(model: torch.nn.Module,
                train_dataloader: Iterable, val_dataloader: Iterable,
                optimizer: torch.optim.Optimizer, device: torch.device,
                output_dir: str, patience: int, scaler: torch.cuda.amp.GradScaler,
                name_of_run: str, criterion, args):

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(args.num_epochs), desc=name_of_run):

        # Adjust learning rate
        current_lr = lr_sched.adjust_learning_rate(optimizer, epoch, args)

        # Training loop
        model.train()
        train_loss = 0.0

        for inputs, targets, _ in tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{args.num_epochs}] Training"):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                targets = targets.squeeze()
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # Validation loop
        model.eval()
        val_loss, all_targets, all_predictions = 0.0, [], []

        with torch.no_grad():
            for inputs, targets, _ in tqdm(val_dataloader, desc=f"Epoch [{epoch+1}/{args.num_epochs}] Validation"):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    targets = targets.squeeze()
                    loss = criterion(outputs, targets)
                val_loss += loss.item()

                outputs = torch.nn.functional.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

        val_loss /= len(val_dataloader)
        all_predictions = torch.cat(all_predictions).numpy()
        all_targets = torch.cat(all_targets).numpy()

        val_accuracy = accuracy_score(all_targets, all_predictions)
        val_f1 = f1_score(all_targets, all_predictions, average="weighted")

        # Print logs and update wandb
        print(f"\nEpoch [{epoch+1}/{args.num_epochs}], Learning Rate: {current_lr:.6f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1Score: {val_f1:.4f}\n\n")
        if args.wandb_enabled:
            wandb.log({
                "epoch": epoch + 1,
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_f1score": val_f1
            })

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            misc.save_model(args, output_dir, "checkpoint-best", model)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Save the last checkpoint
    misc.save_model(args, output_dir, "checkpoint-last", model)

    return model


@torch.no_grad()
def evaluate_model_classification(
    model: torch.nn.Module, test_dataloader: Iterable,
    device: torch.device, config: dict, 
    name_of_run: str, metrics_dir):

    model.to(device)
    model.eval()

    with torch.no_grad():

        output_list, target_list, ground_truth, prediction = [], [], [], []
        for _, (samples, targets, _) in enumerate(test_dataloader):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(samples)

            output_list.append(outputs)
            target_list.append(targets)

            output = torch.nn.functional.softmax(outputs, dim=1)
            predictions = torch.argmax(output, dim=1).cpu().numpy()
            prediction.extend(predictions)
            targets_cpu = targets.cpu().numpy()
            ground_truth.extend(targets_cpu)

    all_outputs = torch.cat(output_list, dim=0)
    all_targets = torch.cat(target_list, dim=0)

    eval_accuracy = accuracy_score(ground_truth, prediction)
    eval_precision = precision_score(ground_truth, prediction, average="weighted")
    eval_recall = recall_score(ground_truth, prediction, average="weighted")
    eval_f1score = f1_score(ground_truth, prediction, average="weighted")
    eval_acc1, eval_acc5 = accuracy(all_outputs, all_targets, topk=(1, 5))
    eval_acc1 = eval_acc1.item()
    eval_acc5 = eval_acc5.item()

    print("-"*60)
    print("Accuracy:", eval_accuracy)
    print("Precision:", eval_precision)
    print("Recall:", eval_recall)
    print("F1-Score:", eval_f1score)
    print("Top-1 Accuracy:", eval_acc1)
    print("Top-5 Accuracy:", eval_acc5)

    ### Plot classification report
    label_dict = config["label_dict"]
    label_dict_reverse = config["label_dict_reverse"]
    label_dict_reverse = {int(k): v for k, v in label_dict_reverse.items()}

    ground_truth = [*map(label_dict_reverse.get, ground_truth)]
    prediction = [*map(label_dict_reverse.get, prediction)]

    cls_report = classification_report(ground_truth, prediction)
    print(cls_report)
    print("-"*60)

    ### Plot confusion matrix
    cmtx = pd.DataFrame(
        confusion_matrix(ground_truth, prediction, labels=list(label_dict.keys())), 
        index=list(label_dict.keys()),
        columns=list(label_dict.keys())
    )
    print(cmtx)
    print("-"*60)

    ### Save results
    save_results(metrics_dir, name_of_run, eval_accuracy, eval_precision, eval_recall, eval_f1score, eval_acc1, eval_acc5, cls_report, cmtx)

    return eval_accuracy, eval_precision, eval_recall, eval_f1score, eval_acc1, eval_acc5


''' Segmentation '''

def training_model_segmentation(model: torch.nn.Module,
                train_dataloader: Iterable, val_dataloader: Iterable,
                optimizer: torch.optim.Optimizer, device: torch.device,
                num_classes: int, output_dir: str, patience: int,
                scaler: torch.cuda.amp.GradScaler, name_of_run: str,
                criterion, class_weights, args):

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(args.num_epochs), desc=name_of_run):

        # Adjust learning rate
        current_lr = lr_sched.adjust_learning_rate(optimizer, epoch, args)

        # Training loop
        model.train()
        train_loss = 0.0
        for inputs, labels, _ in tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{args.num_epochs}] Training"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # Validation loop
        model.eval()
        val_loss, val_iou = 0.0, 0.0

        with torch.no_grad():
            for inputs, labels, _ in tqdm(val_dataloader, desc=f"Epoch [{epoch+1}/{args.num_epochs}] Validation"):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()

                posterior = torch.softmax(outputs, dim=1)

                # Handle binary vs multi-class segmentation
                if num_classes == 2:
                    prediction = torch.argmax(posterior, dim=1, keepdim=True)
                    tp, fp, fn, tn = smp.metrics.get_stats(prediction, labels.type(torch.int64), mode='binary')
                    val_iou += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise").item()
                else:
                    prediction = torch.argmax(posterior, dim=1)
                    # For multi-class, we need to one-hot encode or use proper shape
                    tp, fp, fn, tn = smp.metrics.get_stats(
                        prediction.unsqueeze(1),
                        labels.type(torch.int64),
                        mode='multiclass',
                        num_classes=num_classes
                    )
                    val_iou += smp.metrics.iou_score(tp, fp, fn, tn, reduction="weighted-imagewise", class_weights=class_weights).item()

        val_loss /= len(val_dataloader)
        val_iou /= len(val_dataloader)

        # Print logs and update wandb
        print(f"\nEpoch [{epoch+1}/{args.num_epochs}], Learning Rate: {current_lr:.6f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}\n\n")
        if args.wandb_enabled:
            wandb.log({
                "epoch": epoch + 1,
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_iou
            })

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            misc.save_model(args, output_dir, "checkpoint-best", model)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Save the last checkpoint
    misc.save_model(args, output_dir, "checkpoint-last", model)

    return model


@torch.no_grad()
def evaluate_model_segmentation(
    model: torch.nn.Module, test_dataloader: Iterable,
    device: torch.device, output_dir: str,
    config: dict, class_weights, args):

    model.to(device)
    model.eval()

    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)

    pixel_iou, pixel_accuracy, pixel_precision, pixel_recall, pixel_dice = [], [], [], [], []
    object_precision, object_recall, object_f1 = [], [], []

    num_classes = config["num_classes"]

    with torch.no_grad():
        for _, (inputs, labels, filename) in enumerate(tqdm(test_dataloader)):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)

            posterior = torch.softmax(outputs, dim=1)  # (B, num_classes, H, W)

            # Handle binary vs multi-class segmentation
            if num_classes == 2:
                prediction = torch.argmax(posterior, dim=1, keepdim=True)  # (B, 1, H, W)

                cv2.imwrite(
                    os.path.join(output_dir, "predictions", filename[0]),
                    prediction.cpu().numpy()[0].astype(np.uint8)
                )

                tp, fp, fn, tn = smp.metrics.get_stats(prediction, labels.type(torch.int64), mode='binary')

                pixel_iou.append(smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise", class_weights=class_weights).item())
                pixel_accuracy.append(smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro-imagewise", class_weights=class_weights).item())
                pixel_recall.append(smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise", class_weights=class_weights).item())
                pixel_precision.append(smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise", class_weights=class_weights).item())

                current_dice = (2 * tp) / ((2 * tp) + fp + fn + 1e-8)
                pixel_dice.append(current_dice.item())

                # For object metrics
                pred_np = prediction.cpu().numpy()[0].squeeze()
                labels_np = labels.cpu().numpy()[0].squeeze()

            else:
                # Get class predictions (B, H, W)
                prediction = torch.argmax(posterior, dim=1)

                cv2.imwrite(
                    os.path.join(output_dir, "predictions", filename[0]),
                    prediction.cpu().numpy()[0].astype(np.uint8)
                )
                
                # Ensure labels are in correct format (B, H, W) with class indices
                if labels.dim() == 4:
                    if labels.shape[1] == num_classes:
                        # Labels are one-hot encoded, convert to indices
                        labels_indices = torch.argmax(labels, dim=1)
                    elif labels.shape[1] == 1:
                        # Labels are (B, 1, H, W), squeeze to (B, H, W)
                        labels_indices = labels.squeeze(1)
                    else:
                        raise ValueError(f"Unexpected labels shape: {labels.shape}")
                else:
                    # Labels are already (B, H, W)
                    labels_indices = labels

                tp, fp, fn, tn = smp.metrics.get_stats(
                    prediction.long(),
                    labels_indices.type(torch.int64),
                    mode='multiclass',
                    num_classes=num_classes
                )

                pixel_iou.append(smp.metrics.iou_score(tp, fp, fn, tn, reduction="weighted-imagewise", class_weights=class_weights).item())
                pixel_accuracy.append(smp.metrics.accuracy(tp, fp, fn, tn, reduction="weighted-imagewise", class_weights=class_weights).item())
                pixel_recall.append(smp.metrics.recall(tp, fp, fn, tn, reduction="weighted-imagewise", class_weights=class_weights).item())
                pixel_precision.append(smp.metrics.precision(tp, fp, fn, tn, reduction="weighted-imagewise", class_weights=class_weights).item())

                current_dice = (2 * tp) / ((2 * tp) + fp + fn + 1e-8)
                pixel_dice.append(current_dice.mean().item())

                # For object metrics
                pred_np = prediction.cpu().numpy()[0]
                labels_np = labels_indices.cpu().numpy()[0]

            metrics = compute_object_metrics(labels_np, pred_np, iou_threshold=0.5, num_classes=num_classes)
            object_precision.append(metrics['precision'])
            object_recall.append(metrics['recall'])
            object_f1.append(metrics['f1_score'])

    pixel_iou = np.nanmean(pixel_iou)
    pixel_accuracy = np.nanmean(pixel_accuracy)
    pixel_precision = np.nanmean(pixel_precision)
    pixel_recall = np.nanmean(pixel_recall)
    pixel_dice = np.nanmean(pixel_dice)
    object_precision = np.nanmean(object_precision)
    object_recall = np.nanmean(object_recall)
    object_f1 = np.nanmean(object_f1)

    print("-" * 60)
    print("Pixel IoU:", pixel_iou)
    print("Pixel Accuracy:", pixel_accuracy)
    print("Pixel Precision:", pixel_precision)
    print("Pixel Recall:", pixel_recall)
    print("Pixel Dice:", pixel_dice)
    print("Object Precision:", object_precision)
    print("Object Recall:", object_recall)
    print("Object F1 Score:", object_f1)
    print("-" * 60)

    return pixel_iou, pixel_accuracy, pixel_recall, pixel_precision, pixel_dice, object_precision, object_recall, object_f1
