import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from typing import Iterable
# import wandb
import json
from datetime import datetime

import segmentation_models_pytorch as smp
import torch
from torch.autograd import Variable


''' Classification '''

def training_model_classification(model: torch.nn.Module,
                train_dataloader: Iterable, val_dataloader: Iterable,
                optimizer: torch.optim.Optimizer, device: torch.device,
                epoch: int, num_classes: int, output_dir: str,
                criterion, args):
    
    # Early stopping initialization
    if not hasattr(args, 'best_val_loss'):
        args.best_val_loss = float('inf')
        args.epochs_no_improve = 0
        args.patience = args.patience if hasattr(args, 'patience') else 5  # Default patience of 5 epochs
    
    # Training loop
    model.train()
    train_loss = 0.0

    for inputs, labels, _ in tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{args.num_epochs}] Training"):
        inputs = Variable(inputs.type(torch.FloatTensor)).to(device)
        if num_classes == 2:
            labels = Variable(labels.type(torch.FloatTensor)).to(device)
        else:
            labels = Variable(labels.type(torch.LongTensor)).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        if num_classes == 2:
            outputs = outputs.view(-1)
        else:
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
            if num_classes == 2:
                labels = Variable(labels.type(torch.FloatTensor)).to(device)
            else:
                labels = Variable(labels.type(torch.LongTensor)).to(device)

            outputs = model(inputs)
            if num_classes == 2:
                outputs = outputs.view(-1)
            else:
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
    # if args.wandb_enabled:
    #     wandb.log({
    #         "epoch": epoch + 1,
    #         "train_loss": train_loss,
    #         "val_loss": val_loss,
    #         "val_accuracy": val_accuracy
    #     })

    # Save finetuned model if validation loss improves
    if val_loss < args.best_val_loss:
        print(f"Validation loss improved from {args.best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
        args.best_val_loss = val_loss
        args.epochs_no_improve = 0
        # Save the best model
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
    else:
        args.epochs_no_improve += 1
        print(f"No improvement in validation loss for {args.epochs_no_improve} epochs. Best: {args.best_val_loss:.4f}")
        
        # Check early stopping condition
        if args.epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            # Load the best model
            model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth")))
            return model, True  # Return True to indicate early stopping

    # Save checkpoint every 2 epochs or on first epoch
    if ((epoch + 1) % 2 == 0) or (epoch == 0):
        torch.save(model.state_dict(), os.path.join(output_dir, f"checkpoint_{epoch+1}.pth"))

    return model, False  # Return False to indicate no early stopping


@torch.no_grad()
def evaluate_model_classification(model: torch.nn.Module,
                   test_dataloader: Iterable, device: torch.device,
                   output_dir: str, config: dict,
                   model_name: str, dataset_name: str, checkpoint_name: str):

    if model is not None:
        print(f"Finetuned model is provided and using that for evaluation!")
    else:
        # Find the latest checkpoint in the output directory
        checkpoints = [f for f in os.listdir(output_dir) if f.startswith("checkpoint_") and f.endswith(".pth")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))
            model_path = os.path.join(output_dir, latest_checkpoint)
            print(f"Loading finetuned model from the latest checkpoint: {model_path}")
            model = torch.load(model_path)
        elif model is None:
            raise ValueError("No checkpoints found in the output directory and no model provided.")
    
    model.to(device)
    model.eval()

    with torch.no_grad():

        ground_truth, prediction = [] ,[]
        for _, (inputs, targets, _) in enumerate(tqdm(test_dataloader)):

            inputs = Variable(inputs.type(torch.FloatTensor)).to(device)

            output = model(inputs)

            if config["num_classes"] == 2:
                posterior = output.cpu().detach().numpy()
                output = np.where(posterior > 0.5, 1, 0)
                prediction.append(output[0][0])
            else:
                label = output.cpu().detach().numpy()
                prediction.append(np.argmax(label))

            targets = targets.cpu().detach().numpy()
            ground_truth.append(targets[0])

    
    print("-"*60)
    accuracy = accuracy_score(ground_truth, prediction)
    precision = precision_score(ground_truth, prediction, average="weighted")
    recall = recall_score(ground_truth, prediction, average="weighted")
    f1 = f1_score(ground_truth, prediction, average="weighted")

    ### Plot classification report
    label_dict = config["label_dict"]
    label_dict_reverse = config["label_dict_reverse"]
    label_dict_reverse = {int(k): v for k, v in label_dict_reverse.items()}

    ground_truth = [*map(label_dict_reverse.get, ground_truth)]
    prediction = [*map(label_dict_reverse.get, prediction)]

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print(classification_report(ground_truth, prediction))
    print("-"*60)

    ### Plot confusion matrix
    cmtx = pd.DataFrame(
        confusion_matrix(ground_truth, prediction, labels=list(label_dict.keys())), 
        index=list(label_dict.keys()),
        columns=list(label_dict.keys())
    )
    
    # Display full confusion matrix without truncation
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print("\nConfusion Matrix:")
        print(cmtx)
    print("-"*60)

    return accuracy, precision, recall, f1


''' Segmentation '''

def training_model_segmentation(model: torch.nn.Module,
                train_dataloader: Iterable, val_dataloader: Iterable,
                optimizer: torch.optim.Optimizer, device: torch.device,
                epoch: int, num_classes: int, output_dir: str,
                criterion, args):

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
    # if args.wandb_enabled:
    #     wandb.log({
    #         "epoch": epoch + 1,
    #         "train_loss": train_loss,
    #         "val_loss": val_loss,
    #         "val_iou": val_iou
    #     })

    # Save finetuned model
    if ((epoch + 1) % 2 == 0) or (epoch == 0):
        torch.save(model.state_dict(), os.path.join(output_dir, f"checkpoint_{epoch+1}.pth"))

    return model


@torch.no_grad()
def evaluate_model_segmentation(model: torch.nn.Module,
                   test_dataloader: Iterable, device: torch.device,
                   output_dir: str, config: dict):

    if model is not None:
        print(f"Finetuned model is provided and using that for evaluation!")
    else:
        # Find the latest checkpoint in the output directory
        checkpoints = [f for f in os.listdir(output_dir) if f.startswith("checkpoint_") and f.endswith(".pth")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))
            model_path = os.path.join(output_dir, latest_checkpoint)
            print(f"Loading finetuned model from the latest checkpoint: {model_path}")
            model = torch.load(model_path)
        elif model is None:
            raise ValueError("No checkpoints found in the output directory and no model provided.")
    
    model.to(device)
    model.eval()

    with torch.no_grad():

        pixel_iou, pixel_accuracy, pixel_precision, pixel_recall, pixel_dice = [], [], [], [], []

        for _, (inputs, labels, _) in enumerate(tqdm(test_dataloader)):

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
            tp, fp, fn, tn = smp.metrics.get_stats(prediction, labels.type(torch.int64), mode='binary')
            current_dice_coff = (2 * tp) / ((2* tp) + fp + fn)

            pixel_iou.append(smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise").item())
            pixel_accuracy.append(smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro-imagewise").item())
            pixel_recall.append(smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise").item())
            pixel_precision.append(smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise").item())
            pixel_dice.append(current_dice_coff.item())


    print("-"*60)
    print("Pixel IoU:", sum(pixel_iou) / len(pixel_iou))
    print("Pixel Accuracy:", sum(pixel_accuracy) / len(pixel_accuracy))
    print("Pixel Precision:", sum(pixel_recall) / len(pixel_recall))
    print("Pixel Recall:", sum(pixel_precision) / len(pixel_precision))
    print("Pixel Dice", sum(pixel_dice) / len(pixel_dice))

    return sum(pixel_iou) / len(pixel_iou), sum(pixel_accuracy) / len(pixel_accuracy), sum(pixel_recall) / len(pixel_recall), sum(pixel_precision) / len(pixel_precision), sum(pixel_dice) / len(pixel_dice)
