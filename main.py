from pathlib import Path

import torch

from models.cnn import cnn
from models.efficientnet import EfficientNetClassifier
from models.vit import SmallViT
from utils.data import (
    get_cifar10_loaders,
    cnn_train_transform,
    cnn_test_transform,
    ef_train_transform,
    ef_test_transform,
)
from utils.train import train_model, evaluate_model, set_seed
from utils.plot import plot_training_history, calculate_metrics, plot_confusion_matrix

# Chọn một mô hình: "cnn", "efficientnet", "vit".
MODEL_NAME = "cnn"
DATA_ROOT = Path(__file__).resolve().parent / "data"


def run_cnn(device, root):
    cnn_train_loader, cnn_valid_loader, cnn_test_loader, cnn_train_dataset = get_cifar10_loaders(
        train_transform=cnn_train_transform,
        test_transform=cnn_test_transform,
        batch_size=128,
        train_size=45000,
        valid_size=5000,
        root=str(root),
        seed=42,
    )

    cnn_model = cnn().to(device)
    trained_cnn_model, train_losses_cnn, valid_losses_cnn, train_accs_cnn, valid_accs_cnn = train_model(
        model = cnn_model,
        train_loader = cnn_train_loader,
        valid_loader = cnn_valid_loader,
        epochs=50,
        learning_rate=0.001,
        optimizer_name="adamW",
        device=device,
        checkpoint_path="best_cnn_model.pth"
    )

    y_true_cnn, y_pred_cnn = evaluate_model(trained_cnn_model,
                                            cnn_test_loader,
                                            device)

    plot_training_history(train_losses_cnn,
                          valid_losses_cnn,
                          train_accs_cnn,
                          valid_accs_cnn)
    calculate_metrics(y_true_cnn, y_pred_cnn)
    plot_confusion_matrix(y_true_cnn, y_pred_cnn)


def run_efficientnet(device, root):
    ef_train_loader, ef_valid_loader, ef_test_loader, ef_train_dataset = get_cifar10_loaders(
        train_transform=ef_train_transform,
        test_transform=ef_test_transform,
        batch_size=32,
        train_size=45000,
        valid_size=5000,
        root=str(root),
        seed=42,
    )

    ef_model = EfficientNetClassifier().to(device)
    trained_ef_model, train_losses_ef, valid_losses_ef, train_accs1_ef, valid_accs1_ef = train_model(
        model = ef_model,
        train_loader = ef_train_loader,
        valid_loader = ef_valid_loader,
        epochs = 50,
        learning_rate = 0.001,
        optimizer_name = "adam",
        device = device,
        checkpoint_path = "best_ef_model.pth"
        )

    y_true_ef, y_pred_ef = evaluate_model(model = trained_ef_model,
                                      test_loader = ef_test_loader,
                                      device = device)

    plot_training_history(train_losses_ef,
                          valid_losses_ef,
                          train_accs1_ef,
                          valid_accs1_ef)
    calculate_metrics(y_true_ef, y_pred_ef)
    plot_confusion_matrix(y_true_ef, y_pred_ef)


def run_vit(device, root):
    cnn_train_loader, cnn_valid_loader, cnn_test_loader, cnn_train_dataset = get_cifar10_loaders(
        train_transform=cnn_train_transform,
        test_transform=cnn_test_transform,
        batch_size=128,
        train_size=45000,
        valid_size=5000,
        root=str(root),
        seed=42,
    )

    vit_model = SmallViT(
        image_size=32,
        patch_size=4,
        dim=128,
        depth=4,
        heads=4,
        mlp_dim=512,
        num_classes=10,
        dropout=0.1
    ).to(device)

    trained_vit_model, train_losses_vit, valid_losses_vit, train_accs_vit, valid_accs_vit = train_model(
        model=vit_model,
        train_loader=cnn_train_loader,
        valid_loader=cnn_valid_loader,
        epochs=100,
        learning_rate=0.001,
        optimizer_name="adamW",
        device=device,
        checkpoint_path="best_vit_model.pth"
    )

    # Hàm train_model của bạn đã nạp lại checkpoint tốt nhất.
    y_true_vit, y_pred_vit = evaluate_model(
        model=trained_vit_model,
        test_loader=cnn_test_loader,
        device=device
    )

    plot_training_history(
        train_losses_vit,
        valid_losses_vit,
        train_accs_vit,
        valid_accs_vit
    )

    calculate_metrics(y_true_vit, y_pred_vit)
    plot_confusion_matrix(y_true_vit, y_pred_vit)


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if MODEL_NAME == "cnn":
        run_cnn(device, DATA_ROOT)
    elif MODEL_NAME == "efficientnet":
        run_efficientnet(device, DATA_ROOT)
    elif MODEL_NAME == "vit":
        run_vit(device, DATA_ROOT)
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}")


if __name__ == "__main__":
    main()
