from dataclasses import dataclass

import torchvision.transforms as image_transform
import pytorchvideo.transforms as video_transform


@dataclass
class TransformVideoInfo:
    frames_per_clip: int = 8
    min_shorter_side_size: int = 256
    max_shorter_side_size: int = 320
    val_shorter_side_size: int = 256
    crop_size: int = 224


@dataclass
class TransformImageInfo:
    resize_size: int = 256
    crop_size: int = 224


def transform_video(
        trans_video_info: TransformVideoInfo
    ):
    """transform for video clips

    Args:
        trans_video_info (TransformVideoInfo): information for video transform

    Returns:
        Tuple[pytorchvideo.transforms]: train and val transforms
    """

    train_transform = image_transform.Compose(
        [
            video_transform.ApplyTransformToKey(
                key="video",
                transform=image_transform.Compose(
                    [
                        video_transform.UniformTemporalSubsample(
                            trans_video_info.frames_per_clip
                        ),
                        image_transform.Lambda(lambda x: x / 255.0),
                        video_transform.Normalize(
                            [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]
                        ),
                        video_transform.RandomShortSideScale(
                            min_size=trans_video_info.min_shorter_side_size,
                            max_size=trans_video_info.max_shorter_side_size
                        ),
                        image_transform.RandomCrop(train_transform.crop_size),
                        image_transform.RandomHorizontalFlip(),
                    ]
                ),
            ),
            video_transform.RemoveKey("audio"),
        ]
    )

    val_transform = image_transform.Compose(
        [
            video_transform.ApplyTransformToKey(
                key="video",
                transform=image_transform.Compose(
                    [
                        video_transform.UniformTemporalSubsample(
                            trans_video_info.frames_per_clip
                        ),
                        image_transform.Lambda(lambda x: x / 255.0),
                        video_transform.Normalize(
                            [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]
                        ),
                        video_transform.ShortSideScale(
                            trans_video_info.val_shorter_side_size
                        ),
                        image_transform.CenterCrop(trans_video_info.crop_size),
                    ]
                ),
            ),
            video_transform.RemoveKey("audio"),
        ]
    )

    return train_transform, val_transform


def transform_image(
        trans_image_info: TransformImageInfo
):
    """transform for images

    Args:
        trans_image_info (TransformImageInfo): information for image transform

    Returns:
        Tuple[torchvision.transforms]: train and val transforms
    """

    train_transform = image_transform.Compose(
        [
            image_transform.RandomResizedCrop(trans_image_info.crop_size),
            image_transform.RandomHorizontalFlip(),
            image_transform.ToTensor(),
            image_transform.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )

    val_transform = image_transform.Compose(
        [
            image_transform.Resize(trans_image_info.resize_size),
            image_transform.CenterCrop(trans_image_info.crop_size),
            image_transform.ToTensor(),
            image_transform.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )

    return train_transform, val_transform
