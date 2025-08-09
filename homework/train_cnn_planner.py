# train_cnn_planner.py
"""
Thin wrapper to train the cnn_planner using the existing training routine
(defined in train_transformer_planner.py).
"""

from .train_transformer_planner import train
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train CNN Planner (wrapper)")
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--use_augmentation", action="store_true", default=True)
    parser.add_argument("--longitudinal_weight", type=float, default=1.2)
    parser.add_argument("--lateral_weight", type=float, default=0.8)
    parser.add_argument("--n_waypoints", type=int, default=3)
    parser.add_argument("--device_override", type=str, default=None)
    args = parser.parse_args()

    # Call the shared train routine but force model_name to "cnn_planner"
    train_kwargs = vars(args)
    train_kwargs["model_name"] = "cnn_planner"
    # The shared train() expects some model-specific kwargs; cnn_planner uses only n_waypoints
    train(**train_kwargs)

if __name__ == "__main__":
    main()
