
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
from timm.utils import accuracy
from tqdm import tqdm
from typing import Iterable
import wandb

import segmentation_models_pytorch as smp
import torch
from torch.autograd import Variable

import utils.lr_sched as lr_sched
from utils.metrics import get_object_level_metrics
import utils.misc as misc


''' Classification '''

def save_results(output_dir, name_of_run, eval_accuracy, eval_precision, eval_recall, eval_f1score, eval_acc1, eval_acc5, cls_report, cmtx):

    file1 = open(os.path.join(output_dir, "results.txt"), "a")
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
                output_dir: str, patience: int,
                name_of_run: str, criterion, args):

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(args.num_epochs), desc=name_of_run):

        # Training loop
        model.train()
        train_loss = 0.0

        for inputs, targets, _ in tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{args.num_epochs}] Training"):
            inputs = Variable(inputs.type(torch.FloatTensor)).to(device)
            targets = Variable(targets.type(torch.LongTensor)).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # Validation loop
        model.eval()
        val_loss, ground_truth, prediction = 0.0, [], []

        with torch.no_grad():
            for inputs, targets, _ in tqdm(val_dataloader, desc=f"Epoch [{epoch+1}/{args.num_epochs}] Validation"):
                inputs = Variable(inputs.type(torch.FloatTensor)).to(device)
                targets = Variable(targets.type(torch.LongTensor)).to(device)

                outputs = model(inputs)
                targets = targets.squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                output = torch.nn.functional.softmax(outputs, dim=1)
                predictions = torch.argmax(output, dim=1).cpu().numpy()
                prediction.extend(predictions)
                targets_cpu = targets.cpu().numpy()
                ground_truth.extend(targets_cpu)

        val_loss /= len(val_dataloader)
        val_accuracy = accuracy_score(ground_truth, prediction)
        val_f1 = f1_score(ground_truth, prediction, average="weighted")

        # Print logs and update wandb
        print(f"\nEpoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1Score: {val_f1:.4f}\n\n")
        if args.wandb_enabled:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_f1score": val_f1
            })

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if "vit" in args.train_model:
                misc.save_model(args, output_dir, "checkpoint-best", model)
            else:
                torch.save(model.state_dict(), os.path.join(output_dir, f"checkpoint-best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")

    # Save the last checkpoint
    if "vit" in args.train_model:
        misc.save_model(args, output_dir, "checkpoint-last", model)
    else:
        torch.save(model.state_dict(), os.path.join(output_dir, f"checkpoint-last.pth"))

    return model


'''
def training_model_classification_vit(model: torch.nn.Module,
                train_dataloader: Iterable, val_dataloader: Iterable,
                optimizer: torch.optim.Optimizer, device: torch.device,
                output_dir: str, patience: int, name_of_run: str,
                criterion, loss_scaler, config, args):

    best_val_loss = float('inf')
    patience_counter = 0
    accum_iter = args.accum_iter
    print_freq = 1

    # Debug: Print initial learning rate and model parameters
    print(f"Initial LR: {optimizer.param_groups[0]['lr']}")
    print(f"Accumulation iterations: {accum_iter}")
    
    # Check if model parameters require gradients
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params} / {total_params}")

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Initial LR: {optimizer.param_groups[0]['lr']}")
    
    for epoch in tqdm(range(args.num_epochs), desc=name_of_run):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Training loop - SIMPLIFIED
        model.train()
        train_loss = 0.0
        
        for batch_idx, (samples, targets, _) in enumerate(train_dataloader):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            if targets.dim() > 1:
                targets = targets.squeeze()
            
            optimizer.zero_grad()
            
            # Forward pass WITHOUT autocast for testing
            outputs = model(samples)
            loss = criterion(outputs, targets)
            
            # Simple backward pass WITHOUT gradient scaling
            loss.backward()
            
            # Check gradients before optimizer step
            total_grad_norm = 0
            grad_count = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
                    grad_count += 1
            
            if grad_count > 0:
                total_grad_norm = total_grad_norm ** 0.5
                
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Loss={loss.item():.4f}, Grad_norm={total_grad_norm:.4f}, LR={optimizer.param_groups[0]['lr']:.8f}")
                
                # Print some parameter values to ensure they're changing
                param_sample = next(iter(model.parameters()))
                print(f"  Sample parameter mean: {param_sample.mean().item():.6f}")
        
        train_loss /= min(len(train_dataloader), 6)
        
        # Simple validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        ground_truth, prediction = [], []
        with torch.no_grad():
            for batch_idx, (samples, targets, _) in enumerate(val_dataloader):
                samples = samples.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                if targets.dim() > 1:
                    targets = targets.squeeze()
                
                outputs = model(samples)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                output = torch.nn.functional.softmax(outputs, dim=1)
                predictions = torch.argmax(output, dim=1).cpu().numpy()
                prediction.extend(predictions)
                targets_cpu = targets.cpu().numpy()
                ground_truth.extend(targets_cpu)
        
        val_loss /= min(len(val_dataloader), 3)
        val_acc = 100. * correct / total if total > 0 else 0

        print(classification_report(ground_truth, prediction))
        print("-"*60)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

    
    return model




def training_model_classification_vit(model: torch.nn.Module,
                train_dataloader: Iterable, val_dataloader: Iterable,
                optimizer: torch.optim.Optimizer, device: torch.device,
                output_dir: str, patience: int, name_of_run: str,
                criterion, loss_scaler, config, args):

    # best_val_loss = float('inf')
    # patience_counter = 0
    # accum_iter = args.accum_iter
    # print_freq = 1

    # for epoch in tqdm(range(args.num_epochs), desc=name_of_run):

    #     # Training loop
    #     metric_logger = misc.MetricLogger(delimiter="  ")
    #     metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #     header = 'Training Epoch: [{}]'.format(epoch)

    #     model.train(True)
    #     optimizer.zero_grad()

    #     for data_iter_step, (samples, targets, _) in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)):

    #         if data_iter_step % accum_iter == 0:
    #             lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(train_dataloader) + epoch, args)

    #         samples = samples.to(device, non_blocking=True)
    #         targets = targets.to(device, non_blocking=True)

    #         with torch.cuda.amp.autocast():
    #             outputs = model(samples)
    #             loss = criterion(outputs, targets)

    #         loss_value = loss.item()

    #         if not math.isfinite(loss_value):
    #             print("Loss is {}, stopping training".format(loss_value))
    #             sys.exit(1)

    #         loss /= accum_iter
    #         loss_scaler(loss, optimizer, clip_grad=args.max_norm,
    #                     parameters=model.parameters(), create_graph=False,
    #                     update_grad=(data_iter_step + 1) % accum_iter == 0)
    #         if (data_iter_step + 1) % accum_iter == 0:
    #             optimizer.zero_grad()

    #         torch.cuda.synchronize()

    #         metric_logger.update(loss=loss_value)
    #         min_lr = 10.
    #         max_lr = 0.
    #         for group in optimizer.param_groups:
    #             min_lr = min(min_lr, group["lr"])
    #             max_lr = max(max_lr, group["lr"])

    #         metric_logger.update(lr=max_lr)
    #         # Debug: Print gradient norms periodically
    #         if data_iter_step % 10 == 0:
    #             total_grad_norm = 0
    #             for p in model.parameters():
    #                 if p.grad is not None:
    #                     total_grad_norm += p.grad.data.norm(2).item() ** 2
    #             total_grad_norm = total_grad_norm ** 0.5
    #             print(f"Step {data_iter_step}: Loss={loss_value:.4f}, LR={max_lr:.8f}, Grad_norm={total_grad_norm:.4f}")

    #     current_training_loss = metric_logger.meters["loss"].global_avg

    #     # Validation loop
    #     metric_logger = misc.MetricLogger(delimiter="  ")
    #     header = 'Validation Epoch: [{}]'.format(epoch)

    #     ground_truth, prediction = [], []
    #     model.eval()

    #     with torch.no_grad():

    #         for data_iter_step, (samples, targets, _) in enumerate(metric_logger.log_every(val_dataloader, print_freq, header)):

    #             samples = samples.to(device, non_blocking=True)
    #             targets = targets.to(device, non_blocking=True)

    #             with torch.cuda.amp.autocast():
    #                 output = model(samples)
    #                 loss = criterion(output, targets)

    #             loss_value = loss.item()
    #             if not math.isfinite(loss_value):
    #                 print("Loss is {}, stopping training".format(loss_value))
    #                 sys.exit(1)

    #             output = torch.nn.functional.softmax(output, dim=1)
    #             predictions = torch.argmax(output, dim=1).cpu().numpy()
    #             prediction.extend(predictions)
    #             targets_cpu = targets.cpu().numpy()
    #             ground_truth.extend(targets_cpu)

    #             acc1, acc5 = accuracy(output, targets, topk=(1, 5))

    #             batch_size = samples.shape[0]
    #             metric_logger.update(loss=loss.item())
    #             metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    #             metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    #     current_validation_loss = metric_logger.meters["loss"].global_avg
    #     current_validation_accuracies = {k: meter.global_avg for k, meter in metric_logger.meters.items() if "acc" in k}

    #     if current_validation_loss < best_val_loss:
    #         best_val_loss = current_validation_loss
    #         patience_counter = 0
    #         misc.save_model(
    #             args=args, output_dir=output_dir, model=model, model_without_ddp=model, optimizer=optimizer,
    #             loss_scaler=loss_scaler, epoch="best")
    #     else:
    #         patience_counter += 1
    #         if patience_counter >= patience:
    #             print(f"Early stopping triggered after {epoch + 1} epochs")
    #             break

    #     print(
    #         f"Epoch [{epoch+1}/{args.num_epochs}], \
    #         Train Loss: {current_training_loss:.4f}, \
    #         Val Loss: {current_validation_loss:.4f}, \
    #         Val Accuracy: {accuracy_score(ground_truth, prediction):.4f}, \
    #         Val top 1 Accuracy: {current_validation_accuracies['acc1']:.4f}, \
    #         Val top 5 Accuracy: {current_validation_accuracies['acc5']:.4f}"
    #     )

    #     if args.wandb_enabled:
    #         wandb.log({
    #             "epoch": epoch + 1,
    #             "train_loss": current_training_loss,
    #             "val_loss": current_validation_loss,
    #             "val_accuracy": accuracy_score(ground_truth, prediction),
    #             "val_top_1_accuracy": current_validation_accuracies['acc1'],
    #             "val_top_5_accuracy": current_validation_accuracies['acc5']
    #         })

    # misc.save_model(
    #     args=args, output_dir=output_dir, model=model, model_without_ddp=model, optimizer=optimizer,
    #     loss_scaler=loss_scaler, epoch="last")



    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(args.num_epochs), desc=name_of_run):

        # Training loop
        model.train(True)
        train_losses = []
        # accum_iter = args.accum_iter
        optimizer.zero_grad()

        for data_iter_step, (samples, targets, _) in enumerate(train_dataloader):

            # if data_iter_step % accum_iter == 0:
            #     lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(train_dataloader) + epoch, args)

            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)
                # print("Training loss", loss)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            train_losses.append(loss_value)
            loss.backward()
            optimizer.step()

            # loss /= accum_iter
            # loss_scaler(loss, optimizer, clip_grad=args.max_norm,
            #             parameters=model.parameters(), create_graph=False,
            #             update_grad=True)
            # if (data_iter_step + 1) % accum_iter == 0:
            #     optimizer.zero_grad()
            
            # min_lr = 10.
            # max_lr = 0.
            # for group in optimizer.param_groups:
            #     min_lr = min(min_lr, group["lr"])
            #     max_lr = max(max_lr, group["lr"])

            # torch.cuda.synchronize()

        current_training_loss = sum(train_losses) / len(train_losses)

        # Validation loop
        model.eval()
        val_losses = []
        ground_truth, prediction = [], []
        
        with torch.no_grad():
            for samples, targets, _ in val_dataloader:
                samples = samples.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = model(samples)
                    loss = criterion(outputs, targets)
                    # print("Validation loss", loss)

                loss_value = loss.item()

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

                val_losses.append(loss_value)

                output = torch.nn.functional.softmax(outputs, dim=1)
                predictions = torch.argmax(output, dim=1).cpu().numpy()
                prediction.extend(predictions)
                targets_cpu = targets.cpu().numpy()
                ground_truth.extend(targets_cpu)

        current_validation_loss = sum(val_losses) / len(val_losses)

        # ### Plot classification report
        # label_dict = config["label_dict"]
        # label_dict_reverse = config["label_dict_reverse"]
        # label_dict_reverse = {int(k): v for k, v in label_dict_reverse.items()}

        # ground_truth = [*map(label_dict_reverse.get, ground_truth)]
        # prediction = [*map(label_dict_reverse.get, prediction)]

        # print(classification_report(ground_truth, prediction))
        # print("-"*60)

        # ### Plot confusion matrix
        # cmtx = pd.DataFrame(
        #     confusion_matrix(ground_truth, prediction, labels=list(label_dict.keys())), 
        #     index=list(label_dict.keys()),
        #     columns=list(label_dict.keys())
        # )
        # print(cmtx)
        # print("-"*60)

        print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {current_training_loss:.4f}, Val Loss: {current_validation_loss:.4f}, Val Accuracy: {accuracy_score(ground_truth, prediction):.4f}")
        if args.wandb_enabled:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": current_training_loss,
                "val_loss": current_validation_loss,
                "val_accuracy": accuracy_score(ground_truth, prediction)
            })

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
                break

    misc.save_model(
        args=args, output_dir=output_dir, model=model, model_without_ddp=model, optimizer=optimizer,
        loss_scaler=loss_scaler, epoch="last")

    return model
    '''


@torch.no_grad()
def evaluate_model_classification(
    model: torch.nn.Module, test_dataloader: Iterable,
    device: torch.device, result_csv_path: str, balance_data: str,
    config: dict, pretraining_configuration: str, output_dir: str,
    name_of_run: str, no_of_samples: int, args):

    model.to(device)
    model.eval()

    with torch.no_grad():

        output_list, target_list, ground_truth, prediction = [], [], [], []
        for data_iter_step, (samples, targets, _) in enumerate(test_dataloader):
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
    save_results(output_dir, name_of_run, eval_accuracy, eval_precision, eval_recall, eval_f1score, eval_acc1, eval_acc5, cls_report, cmtx)

    if os.path.exists(result_csv_path):
        result_df = pd.read_csv(result_csv_path)
    else:
        result_df = pd.DataFrame(columns=[
            "Downstream Task", "Training type",
            "Train Model", "Pre-training configuration",
            "balance_data", "no_of_training_samples",
            "Accuracy", "Precision",
            "Recall", "F1-Score",
            "Top-1 Accuracy", "Top-5 Accuracy"
        ])
    current_result = [
        args.dataset, args.which_pretraining,
        args.train_model, pretraining_configuration, balance_data, no_of_samples,
        round(eval_accuracy, 4), round(eval_precision, 4),
        round(eval_recall, 4), round(eval_f1score, 4),
        round(eval_acc1, 4), round(eval_acc5, 4)
    ]
    result_df.loc[len(result_df)] = current_result
    result_df.to_csv(result_csv_path, index=False)

    return eval_accuracy, eval_precision, eval_recall, eval_f1score, eval_acc1, eval_acc5


''' Segmentation '''

def training_model_segmentation(model: torch.nn.Module,
                train_dataloader: Iterable, val_dataloader: Iterable,
                optimizer: torch.optim.Optimizer, device: torch.device,
                num_classes: int, output_dir: str, patience: int,
                name_of_run: str, criterion, args):

    best_val_loss = float('inf')
    patience_counter = 0

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

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if "vit" in args.train_model:
                misc.save_model(args, output_dir, "checkpoint-best", model)
            else:
                torch.save(model.state_dict(), os.path.join(output_dir, f"checkpoint-best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")

    # Save the last checkpoint
    if "vit" in args.train_model:
        misc.save_model(args, output_dir, "checkpoint-last", model)
    else:
        torch.save(model.state_dict(), os.path.join(output_dir, f"checkpoint-last.pth"))

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
        all_tps, all_fps, all_fns = 0, 0, 0
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
            current_tps, current_fps, current_fns = get_object_level_metrics(labels.cpu().numpy()[0], outputs.cpu().numpy()[0])
            all_tps += current_tps
            all_fps += current_fps
            all_fns += current_fns


    pixel_iou = sum(pixel_iou) / len(pixel_iou)
    pixel_accuracy = sum(pixel_accuracy) / len(pixel_accuracy)
    pixel_precision = sum(pixel_precision) / len(pixel_precision)
    pixel_recall = sum(pixel_recall) / len(pixel_recall)
    pixel_dice = sum(pixel_dice) / len(pixel_dice)
    if all_tps + all_fps > 0:
        object_precision = all_tps / (all_tps + all_fps)
    else:
        object_precision = float("nan")

    if all_tps + all_fns > 0:
        object_recall = all_tps / (all_tps + all_fns)
    else:
        object_recall = float("nan")

    print("-"*60)
    print("Pixel IoU:", pixel_iou)
    print("Pixel Accuracy:", pixel_accuracy)
    print("Pixel Precision:", pixel_precision)
    print("Pixel Recall:", pixel_recall)
    print("Pixel Dice", pixel_dice)
    print("Object Precision:", object_precision)
    print("Object Recall:", object_recall)

    ### Save results
    if os.path.exists(result_csv_path):
        result_df = pd.read_csv(result_csv_path)
    else:
        result_df = pd.DataFrame(columns=[
            "Downstream Task", "Training type",
            "Train Model", "Pre-training configuration",
            "Pixel IoU", "Pixel Accuracy",
            "Pixel Precision", "Pixel Recall",
            "Pixel Dice", "Object Precision", "Object Recall"
        ])
    current_result = [
        args.dataset, args.which_pretraining,
        args.train_model, pretraining_configuration,
        pixel_iou, pixel_accuracy, pixel_precision, pixel_recall,
        pixel_dice, object_precision, object_recall
    ]
    result_df.loc[len(result_df)] = current_result
    result_df.to_csv(result_csv_path, index=False)

    return pixel_iou, pixel_accuracy, pixel_recall, pixel_precision, pixel_dice, object_precision, object_recall
