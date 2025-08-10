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


class CNNPlannerLoss(nn.Module):
    def __init__(
        self,
        longitudinal_weight: float = 1.0,
        lateral_weight: float = 1.0,
        smoothness_weight: float = 0.02,
    ):
        """
        L1-based loss split into longitudinal/lateral components with a small
        smoothness regularizer for CNN planner.
        """
        super().__init__()
        self.longitudinal_weight = longitudinal_weight
        self.lateral_weight = lateral_weight
        self.smoothness_weight = smoothness_weight

    def forward(self, preds, labels, labels_mask):
        # preds/labels: (b, n_waypoints, 2)
        errors = (preds - labels).abs()
        errors_masked = errors * labels_mask.unsqueeze(-1)
        num_valid = labels_mask.sum() + 1e-6

        longitudinal_loss = errors_masked[:, :, 0].sum() / num_valid
        lateral_loss = errors_masked[:, :, 1].sum() / num_valid

        # Smoothness: L1 on consecutive waypoint deltas
        if preds.shape[1] > 1:
            delta_preds = (preds[:, 1:, :] - preds[:, :-1, :]).abs()
            # mask for pairs: both waypoints must be valid
            mask_pairs = (labels_mask[:, 1:] * labels_mask[:, :-1]).unsqueeze(-1)
            smoothness = (delta_preds * mask_pairs).sum() / (mask_pairs.sum() + 1e-6)
        else:
            smoothness = torch.tensor(0.0, device=preds.device)

        total_loss = (self.longitudinal_weight * longitudinal_loss +
                      self.lateral_weight * lateral_loss +
                      self.smoothness_weight * smoothness)
        return total_loss


def train(
    exp_dir: str = "logs",
    model_name: str = "cnn_planner",
    num_epoch: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 2024,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 5,
    use_augmentation: bool = True,
    longitudinal_weight: float = 1.2,
    lateral_weight: float = 0.8,
    n_waypoints: int = 3,
    hidden_dim: int = 64,
    device_override: str = None,
    **kwargs,
):
    # Device selection
    if device_override:
        device = torch.device(device_override)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    # Seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Logging
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = tb.SummaryWriter(log_dir)

    # Model kwargs
    model_kwargs = {
        "n_waypoints": n_waypoints,
        "hidden_dim": hidden_dim,
    }

    model = load_model(model_name, **model_kwargs)
    model = model.to(device)

    print(f"{model_name} loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

    # Data pipelines - CNN planner uses images
    train_transform = "default"  # includes images
    val_transform = "default"

    train_data = load_data(
        "drive_data/train",
        transform_pipeline=train_transform,
        shuffle=True,
        batch_size=batch_size,
        num_workers=8,
    )
    val_data = load_data(
        "drive_data/val",
        transform_pipeline=val_transform,
        shuffle=False,
        batch_size=batch_size,
        num_workers=8,
    )

    print(f"Train batches: {len(train_data)}, Val batches: {len(val_data)}")

    # Loss function
    loss_func = CNNPlannerLoss(
        longitudinal_weight=longitudinal_weight,
        lateral_weight=lateral_weight,
        smoothness_weight=0.02,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Linear warmup + cosine annealing
    try:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
    except Exception:
        warmup_scheduler = None
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, num_epoch - warmup_epochs)
    )

    # AMP scaler for mixed precision
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_metric = PlannerMetric()
    val_metric = PlannerMetric()

    best_val_l1 = float("inf")
    best_epoch = 0
    patience_counter = 0
    patience = 20  # Slightly less patience for CNN

    # Training loop
    for epoch in range(num_epoch):
        train_metric.reset()
        val_metric.reset()
        model.train()
        train_loss = 0.0
        num_train_batches = 0

        for batch in train_data:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # CNN planner uses images
            preds = model(batch["image"])
            waypoints = batch["waypoints"]
            waypoints_mask = batch["waypoints_mask"]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = loss_func(preds, waypoints, waypoints_mask)

            scaler.scale(loss).backward()

            # gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            # Learning rate scheduling
            if warmup_scheduler is not None and epoch < warmup_epochs:
                warmup_scheduler.step()
            elif epoch >= warmup_epochs:
                scheduler.step()

            train_metric.add(preds.detach().cpu(), waypoints.detach().cpu(), waypoints_mask.detach().cpu())

            train_loss += loss.item()
            num_train_batches += 1

        # Validation
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        with torch.inference_mode():
            for batch in val_data:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                preds = model(batch["image"])
                waypoints = batch["waypoints"]
                waypoints_mask = batch["waypoints_mask"]

                loss = loss_func(preds, waypoints, waypoints_mask)
                val_metric.add(preds.detach().cpu(), waypoints.detach().cpu(), waypoints_mask.detach().cpu())
                val_loss += loss.item()
                num_val_batches += 1

        # Compute metrics and logging
        train_metrics = train_metric.compute()
        val_metrics = val_metric.compute()

        epoch_train_loss = train_loss / max(1, num_train_batches)
        epoch_val_loss = val_loss / max(1, num_val_batches)

        current_lr = optimizer.param_groups[0]['lr']

        logger.add_scalar("l1_error/train", train_metrics["l1_error"], epoch)
        logger.add_scalar("l1_error/val", val_metrics["l1_error"], epoch)
        logger.add_scalar("longitudinal_error/train", train_metrics["longitudinal_error"], epoch)
        logger.add_scalar("longitudinal_error/val", val_metrics["longitudinal_error"], epoch)
        logger.add_scalar("lateral_error/train", train_metrics["lateral_error"], epoch)
        logger.add_scalar("lateral_error/val", val_metrics["lateral_error"], epoch)
        logger.add_scalar("loss/train", epoch_train_loss, epoch)
        logger.add_scalar("loss/val", epoch_val_loss, epoch)
        logger.add_scalar("learning_rate", current_lr, epoch)

        epoch_val_l1 = val_metrics["l1_error"]
        epoch_val_long = val_metrics["longitudinal_error"]
        epoch_val_lat = val_metrics["lateral_error"]

        # Save best model
        if epoch_val_l1 < best_val_l1:
            best_val_l1 = epoch_val_l1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), log_dir / f"{model_name}_best.th")
        else:
            patience_counter += 1

        # Periodic autosave
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), log_dir / f"{model_name}_epoch_{epoch+1}.th")

        # Early stopping
        if patience_counter >= patience and epoch > warmup_epochs:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

        # Print progress
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == num_epoch - 1:
            target_status = "✓" if epoch_val_long < 0.3 and epoch_val_lat < 0.45 else "✗"
            print(
                f"Epoch {epoch + 1}/{num_epoch}: "
                f"train_l1={train_metrics['l1_error']:.4f} val_l1={epoch_val_l1:.4f} "
                f"long={epoch_val_long:.4f} lat={epoch_val_lat:.4f} "
                f"loss={epoch_val_loss:.4f} lr={current_lr:.6f} {target_status}"
            )

    # Final evaluation
    print(f"\nTraining completed! Best val L1: {best_val_l1:.4f} at epoch {best_epoch+1}")
    if best_val_l1 < float("inf"):
        model.load_state_dict(torch.load(log_dir / f"{model_name}_best.th", map_location=device))
        print(f"Loaded best model from epoch {best_epoch+1}")

    # Final validation metrics
    val_metric.reset()
    model.eval()
    with torch.inference_mode():
        for batch in val_data:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            preds = model(batch["image"])
            val_metric.add(preds.detach().cpu(), batch["waypoints"].detach().cpu(), batch["waypoints_mask"].detach().cpu())

    final_metrics = val_metric.compute()
    print("\nFinal validation metrics:")
    print(f"  L1 Error: {final_metrics['l1_error']:.4f}")
    print(f"  Longitudinal Error: {final_metrics['longitudinal_error']:.4f}")
    print(f"  Lateral Error: {final_metrics['lateral_error']:.4f}")

    # Check target metrics for CNN planner
    if final_metrics['longitudinal_error'] < 0.3 and final_metrics['lateral_error'] < 0.45:
        print("🎉 Successfully achieved target metrics!")
    else:
        print("⚠️ Target not achieved (long < 0.3, lat < 0.45) — try tuning hyperparameters or training longer")

    # Save final model for grading
    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}_final.th")
    logger.close()

    return best_val_l1


def main():
    parser = argparse.ArgumentParser(description="Train CNN Planner")
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="cnn_planner",
                        choices=["mlp_planner", "transformer_planner", "cnn_planner"])
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--use_augmentation", action="store_true", default=True)
    parser.add_argument("--longitudinal_weight", type=float, default=1.2)
    parser.add_argument("--lateral_weight", type=float, default=0.8)
    parser.add_argument("--n_waypoints", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    
    args = parser.parse_args()
    best_l1 = train(**vars(args))
    print(f"\nFinal best validation L1 error: {best_l1:.4f}")
    print(f"{args.model_name} training completed!")


if __name__ == "__main__":
    main()