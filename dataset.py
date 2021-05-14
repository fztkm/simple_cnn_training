from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


def transform_factory(args, do_crop=True):
    if do_crop:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])
    return transform


def dataset_facory(args):
    """dataset factory

    Args:
        args (argparse): args

    Returns:
        DataLoader: training set loader
        DataLoader: validation set loader
    """

    transform = transform_factory(args)

    if args.dataset_name == "CIFAR10":
        train_set = CIFAR10(root=args.root,
                            train=True,
                            download=True,
                            transform=transform)
        val_set = CIFAR10(root=args.root,
                          train=False,
                          download=True,
                          transform=transform)
        n_classes = 10
    else:
        raise ValueError("invalid args.dataset_name")

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)

    return train_loader, val_loader, n_classes