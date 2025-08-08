import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from .datasets.road_dataset import load_data
from .models import load_model, save_model
from .metrics import PlannerMetric


class PlannerLoss(nn.Module):
    def __init__(self, longitudinal_weight: float = 1.0, lateral_weight: float = 1.0):
        """
        Custom loss for planner that can weight longitudinal vs lateral errors differently
        """
        super().__init__()
        self.longitudinal_weight = longitudinal_weight
        self.lateral_weight = lateral_weight
        
    def forward(self, preds, labels, labels_mask):
        """
        Args:
            preds: (b, n_waypoints, 2) predicted waypoints
            labels: (b, n_waypoints, 2) ground truth waypoints
            labels_mask: (b, n_waypoints) mask for valid waypoints
        
        Returns:
            scalar loss value
        """
        # Compute L1 loss for each dimension
        errors = (preds - labels).abs()
        
        # Apply mask to ignore invalid waypoints
        errors_masked = errors * labels_mask.unsqueeze(-1)
        
        # Separate longitudinal (x) and lateral (y) errors
        longitudinal_error = errors_masked[:, :, 0]
        lateral_error = errors_masked[:, :, 1]
        
        # Compute weighted mean loss
        num_valid = labels_mask.sum() + 1e-6  # avoid division by zero
        
        longitudinal_loss = longitudinal_error.sum() / num_valid
        lateral_loss = lateral_error.sum() / num_valid
        
        total_loss = (self.longitudinal_weight * longitudinal_loss + 
                     self.lateral_weight * lateral_loss)
        
        return total_loss


def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 2024,
    weight_decay: float = 1e-4,
    scheduler_step_size: int = 25,
    scheduler_gamma: float = 0.5,
    use_augmentation: bool = False,
    longitudinal_weight: float = 1.0,
    lateral_weight: float = 1.0,
    **kwargs,
):
    """
    Train the MLP planner model
    
    Args:
        exp_dir: Directory to save logs and checkpoints
        model_name: Name of the model to train (mlp_planner, transformer_planner, cnn_planner)
        num_epoch: Number of training epochs
        lr: Learning rate
        batch_size: Training batch size
        seed: Random seed for reproducibility
        weight_decay: L2 regularization strength
        scheduler_step_size: Epochs between learning rate decay
        scheduler_gamma: Learning rate decay factor
        use_augmentation: Whether to use data augmentation for training
        longitudinal_weight: Weight for longitudinal error in loss
        lateral_weight: Weight for lateral error in loss
        **kwargs: Additional model parameters
    """
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Create directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = tb.SummaryWriter(log_dir)

    # Load model
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    
    print(f"Model loaded: {model_name}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Select appropriate data pipeline based on model type
    if model_name == "cnn_planner":
        train_transform = "aug" if use_augmentation else "default"
        val_transform = "default"
    else:
        # MLP and Transformer planners use state_only (no images)
        train_transform = "aug" if use_augmentation else "state_only"
        val_transform = "state_only"
    
    # Load data
    train_data = load_data(
        "drive_data/train",
        transform_pipeline=train_transform,
        shuffle=True,
        batch_size=batch_size,
        num_workers=2
    )
    
    val_data = load_data(
        "drive_data/val",
        transform_pipeline=val_transform,
        shuffle=False,
        batch_size=batch_size,
        num_workers=2
    )

    print(f"Training with {len(train_data)} batches, validation with {len(val_data)} batches")
    print(f"Using transform: train={train_transform}, val={val_transform}")

    # Create loss function and optimizer
    loss_func = PlannerLoss(
        longitudinal_weight=longitudinal_weight,
        lateral_weight=lateral_weight
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma
    )

    # Metrics tracking
    train_metric = PlannerMetric()
    val_metric = PlannerMetric()
    
    best_val_l1 = float('inf')
    best_epoch = 0

    # Training loop
    for epoch in range(num_epoch):
        # Reset metrics
        train_metric.reset()
        val_metric.reset()
        
        # Training phase
        model.train()
        train_loss = 0.0
        num_train_batches = 0
        
        for batch in train_data:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Extract inputs based on model type
            if model_name == "cnn_planner":
                image = batch["image"]
                preds = model(image)
            else:
                track_left = batch["track_left"]
                track_right = batch["track_right"]
                preds = model(track_left, track_right)
            
            waypoints = batch["waypoints"]
            waypoints_mask = batch["waypoints_mask"]

            # Compute loss
            loss = loss_func(preds, waypoints, waypoints_mask)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # Update metrics
            train_metric.add(preds, waypoints, waypoints_mask)
            
            train_loss += loss.item()
            num_train_batches += 1

        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.inference_mode():
            for batch in val_data:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Extract inputs based on model type
                if model_name == "cnn_planner":
                    image = batch["image"]
                    preds = model(image)
                else:
                    track_left = batch["track_left"]
                    track_right = batch["track_right"]
                    preds = model(track_left, track_right)
                
                waypoints = batch["waypoints"]
                waypoints_mask = batch["waypoints_mask"]

                # Compute loss
                loss = loss_func(preds, waypoints, waypoints_mask)
                
                # Update metrics
                val_metric.add(preds, waypoints, waypoints_mask)
                
                val_loss += loss.item()
                num_val_batches += 1

        # Step the scheduler
        scheduler.step()
        
        # Compute epoch metrics
        train_metrics = train_metric.compute()
        val_metrics = val_metric.compute()
        
        # Extract metrics
        epoch_train_l1 = train_metrics["l1_error"]
        epoch_train_long = train_metrics["longitudinal_error"]
        epoch_train_lat = train_metrics["lateral_error"]
        
        epoch_val_l1 = val_metrics["l1_error"]
        epoch_val_long = val_metrics["longitudinal_error"]
        epoch_val_lat = val_metrics["lateral_error"]
        
        # Average losses
        epoch_train_loss = train_loss / num_train_batches
        epoch_val_loss = val_loss / num_val_batches
        
        current_lr = optimizer.param_groups[0]['lr']

        # Log to tensorboard
        logger.add_scalar("l1_error/train", epoch_train_l1, epoch)
        logger.add_scalar("l1_error/val", epoch_val_l1, epoch)
        logger.add_scalar("longitudinal_error/train", epoch_train_long, epoch)
        logger.add_scalar("longitudinal_error/val", epoch_val_long, epoch)
        logger.add_scalar("lateral_error/train", epoch_train_lat, epoch)
        logger.add_scalar("lateral_error/val", epoch_val_lat, epoch)
        logger.add_scalar("loss/train", epoch_train_loss, epoch)
        logger.add_scalar("loss/val", epoch_val_loss, epoch)
        logger.add_scalar("learning_rate", current_lr, epoch)

        # Save best model based on L1 error
        if epoch_val_l1 < best_val_l1:
            best_val_l1 = epoch_val_l1
            best_epoch = epoch
            # Save best model weights in log directory
            torch.save(model.state_dict(), log_dir / f"{model_name}_best.th")

        # Print progress
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d} / {num_epoch:3d}: "
                f"train_l1={epoch_train_l1:.4f} "
                f"val_l1={epoch_val_l1:.4f} "
                f"long={epoch_val_long:.4f} "
                f"lat={epoch_val_lat:.4f} "
                f"loss={epoch_val_loss:.4f} "
                f"lr={current_lr:.6f}"
            )

    # Final results
    print(f"\nTraining completed!")
    print(f"Best validation L1 error: {best_val_l1:.4f} at epoch {best_epoch + 1}")
    
    # Load best model for final save
    if best_val_l1 < float('inf'):
        model.load_state_dict(torch.load(log_dir / f"{model_name}_best.th", map_location=device))
        print(f"Loaded best model from epoch {best_epoch + 1}")
        
        # Final evaluation
        val_metric.reset()
        model.eval()
        with torch.inference_mode():
            for batch in val_data:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                if model_name == "cnn_planner":
                    image = batch["image"]
                    preds = model(image)
                else:
                    track_left = batch["track_left"]
                    track_right = batch["track_right"]
                    preds = model(track_left, track_right)
                
                waypoints = batch["waypoints"]
                waypoints_mask = batch["waypoints_mask"]
                val_metric.add(preds, waypoints, waypoints_mask)
        
        final_metrics = val_metric.compute()
        print(f"\nFinal validation metrics:")
        print(f"  L1 Error: {final_metrics['l1_error']:.4f}")
        print(f"  Longitudinal Error: {final_metrics['longitudinal_error']:.4f}")
        print(f"  Lateral Error: {final_metrics['lateral_error']:.4f}")
        
        # Check if target metrics are achieved
        if final_metrics['longitudinal_error'] < 0.2 and final_metrics['lateral_error'] < 0.6:
            print("🎉 Successfully achieved target metrics!")
        else:
            print(f"⚠️  Target not achieved (long < 0.2, lat < 0.6)")
    
    # Save final model in root directory for grading
    save_model(model)
    print(f"Final model saved for grading")

    # Also save in log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}_final.th")
    print(f"Model weights saved to {log_dir}")
    
    logger.close()
    
    return best_val_l1


def main():
    parser = argparse.ArgumentParser(description="Train SuperTux Planner Model")

    # Basic training parameters
    parser.add_argument("--exp_dir", type=str, default="logs",
                       help="Directory to save logs and checkpoints")
    parser.add_argument("--model_name", type=str, default="mlp_planner",
                       choices=["mlp_planner", "transformer_planner", "cnn_planner"],
                       help="Name of the model to train")
    parser.add_argument("--num_epoch", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--seed", type=int, default=2024,
                       help="Random seed for reproducibility")
    
    # Optimization parameters
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="L2 regularization strength")
    parser.add_argument("--scheduler_step_size", type=int, default=25,
                       help="Epochs between learning rate decay")
    parser.add_argument("--scheduler_gamma", type=float, default=0.5,
                       help="Learning rate decay factor")
    
    # Data augmentation
    parser.add_argument("--use_augmentation", action="store_true", default=False,
                       help="Whether to use data augmentation")
    
    # Loss weighting
    parser.add_argument("--longitudinal_weight", type=float, default=1.0,
                       help="Weight for longitudinal error in loss")
    parser.add_argument("--lateral_weight", type=float, default=1.0,
                       help="Weight for lateral error in loss")
    
    # Model hyperparameters
    parser.add_argument("--n_track", type=int, default=10,
                       help="Number of track boundary points")
    parser.add_argument("--n_waypoints", type=int, default=3,
                       help="Number of waypoints to predict")
    parser.add_argument("--d_model", type=int, default=64,
                       help="Model dimension for transformer")

    args = parser.parse_args()
    
    # Extract model kwargs based on model type
    if args.model_name == "mlp_planner":
        model_kwargs = {
            "n_track": args.n_track,
            "n_waypoints": args.n_waypoints,
        }
    elif args.model_name == "transformer_planner":
        model_kwargs = {
            "n_track": args.n_track,
            "n_waypoints": args.n_waypoints,
            "d_model": args.d_model,
        }
    elif args.model_name == "cnn_planner":
        model_kwargs = {
            "n_waypoints": args.n_waypoints,
        }
    else:
        model_kwargs = {}
    
    # Remove model kwargs from args dict
    args_dict = vars(args)
    for key in ["n_track", "n_waypoints", "d_model"]:
        args_dict.pop(key, None)
    
    # Train the model
    best_l1 = train(**args_dict, **model_kwargs)
    
    print(f"\nFinal best validation L1 error: {best_l1:.4f}")


if __name__ == "__main__":
    main()