import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


class_names = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}

def predict_image(model, image, label, device):
    model.eval()
    display_image = image
    input_image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_image)
        predicted = torch.argmax(output, dim=1).item()

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(display_image.permute(1, 2, 0))

    ax.set_title(
        f"Predicted: {predicted} ({class_names[predicted]})\n"
        f"Label: {label} ({class_names[label]})"
    )
    ax.axis("off")
    plt.show()

def plot_training_history(train_losses, valid_losses, train_accs, valid_accs):
    epochs = range(1, len(train_losses) + 1)

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, valid_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, valid_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(9, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - CIFAR-10")

    plt.tight_layout()
    plt.show()

def calculate_metrics(y_true, y_pred):
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]
    precision = precision_score(
        y_true, y_pred,
        average=None,
        zero_division=0
    )

    recall = recall_score(
        y_true, y_pred,
        average=None,
        zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred,
        average=None,
        zero_division=0
    )

    print("\nClassification Metrics:")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-score':>10}")
    print("-" * 46)

    for i, class_name in enumerate(class_names):
        print(
            f"{class_name:<12} "
            f"{precision[i]:>10.4f} "
            f"{recall[i]:>10.4f} "
            f"{f1[i]:>10.4f}"
        )

    # Macro average
    print("-" * 46)
    print(
        f"{'Macro Avg':<12} "
        f"{precision_score(y_true, y_pred, average='macro', zero_division=0):>10.4f} "
        f"{recall_score(y_true, y_pred, average='macro', zero_division=0):>10.4f} "
        f"{f1_score(y_true, y_pred, average='macro', zero_division=0):>10.4f}"
    )
