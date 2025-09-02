
import os
import json
from tqdm import tqdm

import torch
from typing import Iterable


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
            for images in train_dataloader:
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
                for images in val_dataloader:
                    images = images.to(device)
                    reconstructions = model(images)
                    val_loss += criterion(reconstructions, images).item()
            val_loss /= len(val_dataloader)

            # Store metrics
            training_metrics["epochs"].append(epoch + 1)
            training_metrics["train_loss"].append(float(train_loss.item()))
            training_metrics["val_loss"].append(float(val_loss))

            # Log to wandb
            if args.wandb_enabled:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                })
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss:.4f}")

            # Save the encoder part of the model
            if ((epoch + 1) % 1 == 0) or (epoch == 0):
                # Save the encoder and decoder part of the model separately
                torch.save(model.encoder.state_dict(), os.path.join(output_dir, f"encoder_epoch_{epoch+1}.pth"))
                torch.save(model.decoder.state_dict(), os.path.join(output_dir, f"decoder_epoch_{epoch+1}.pth"))


    # Save training metrics to JSON file
    metrics_file = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(training_metrics, f, indent=4)

    print(f"Training metrics saved to {metrics_file}")
