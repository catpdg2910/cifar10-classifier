import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(
    model,
    train_loader,
    valid_loader,
    epochs,
    learning_rate,
    optimizer_name,
    device,
    patience = 5,
    checkpoint_path = "best_model.pth"
):
    criterion = nn.CrossEntropyLoss()

    trainable_params = filter(
        lambda p: p.requires_grad,
        model.parameters()
    )

    if optimizer_name == "adam":
        optimizer = optim.Adam(
            trainable_params,
            lr=learning_rate
        )
    elif optimizer_name == "adamW":
        optimizer = optim.AdamW(
            trainable_params,
            lr=learning_rate
        )
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            trainable_params,
            lr=learning_rate
        )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )
    
    
    # Lưu loss của từng epoch
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    best_valid_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):

        # =====================
        # TRAIN
        # =====================
        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total

        # =====================
        # VALIDATION
        # =====================
        model.eval()

        valid_loss = 0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()

        valid_loss /= len(valid_loader)
        valid_acc = 100 * valid_correct / valid_total

        # Lưu loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.2f}% | "
            f"Valid Loss: {valid_loss:.4f}, "
            f"Valid Acc: {valid_acc:.2f}%"
        )

        # =====================
        # EARLY STOPPING
        # =====================
        if valid_loss < best_valid_loss:

            best_valid_loss = valid_loss
            patience_counter = 0

            torch.save(model.state_dict(), checkpoint_path)

        else:
            patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping!")
                break
                
        scheduler.step()

    # Load best model
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )

    return model, train_losses, valid_losses, train_accs, valid_accs


def evaluate_model(model, test_loader, device):
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    y_true = []
    y_pred = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    test_loss /= len(test_loader)
    test_acc = 100 * test_correct / test_total
    print(
        f"Test Loss: {test_loss:.4f}, "
        f"Test Acc: {test_acc:.2f}%"
    )
    return y_true, y_pred
