import torch
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights


weights_b0 = EfficientNet_B0_Weights.DEFAULT
preprocess_b0 = weights_b0.transforms()

ef_train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),

    transforms.Resize(
        (224, 224),
        interpolation=preprocess_b0.interpolation,
        antialias=True
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=preprocess_b0.mean,
        std=preprocess_b0.std
    )
])
ef_test_transform = transforms.Compose([
   transforms.Resize(
        (224, 224),
        interpolation=preprocess_b0.interpolation,
        antialias=True
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=preprocess_b0.mean,
        std=preprocess_b0.std
    )
])

cnn_train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])
cnn_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

def get_cifar10_loaders(
    train_transform,
    test_transform,
    batch_size=128,
    train_size=40000,
    valid_size=10000,
    root="./data",
    seed=42,
    download=True
):
    """
    Tạo DataLoader cho CIFAR-10.

    Parameters
    ----------
    train_transform : torchvision.transforms
        Transform dùng cho tập train.

    test_transform : torchvision.transforms
        Transform dùng cho validation và test.

    batch_size : int
        Kích thước mỗi batch.

    train_size : int
        Số lượng ảnh dùng để train.

    valid_size : int
        Số lượng ảnh dùng để validation.

    root : str
        Đường dẫn lưu CIFAR-10.

    seed : int
        Seed dùng để chia train/validation.

    download : bool
        Tải CIFAR-10 nếu chưa có trong root.
    """

    # ========================================================
    # 1. Tạo generator để chia dataset có thể tái lập
    # ========================================================

    generator = torch.Generator().manual_seed(seed)


    # ========================================================
    # 2. Load CIFAR-10 không có transform
    #
    # Dataset này chỉ dùng để lấy index train/validation.
    # ========================================================

    full_dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=None
    )


    # ========================================================
    # 3. Chia dataset thành train và validation
    #
    # Chỉ random_split MỘT LẦN để đảm bảo không overlap.
    # ========================================================

    train_subset, valid_subset = random_split(
        full_dataset,
        [train_size, valid_size],
        generator=generator
    )

    train_indices = train_subset.indices
    valid_indices = valid_subset.indices


    # ========================================================
    # 4. Tạo dataset train
    #
    # Dùng train_transform.
    # ========================================================

    train_full = datasets.CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=train_transform
    )


    # ========================================================
    # 5. Tạo dataset validation
    #
    # Dùng test_transform để KHÔNG augmentation.
    # ========================================================

    valid_full = datasets.CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=test_transform
    )


    # ========================================================
    # 6. Lấy đúng index đã split ở bước 3
    # ========================================================

    train_dataset = Subset(
        train_full,
        train_indices
    )

    valid_dataset = Subset(
        valid_full,
        valid_indices
    )


    # ========================================================
    # 7. Load test dataset
    #
    # CIFAR-10 test là 10,000 ảnh riêng biệt,
    # không liên quan đến train/validation.
    # ========================================================

    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=test_transform
    )


    # ========================================================
    # 8. Tạo DataLoader
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    # ========================================================
    # 9. Trả về 3 DataLoader
    # ========================================================

    return train_loader, valid_loader, test_loader, train_dataset
