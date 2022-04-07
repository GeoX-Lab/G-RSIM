import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def make_transform(args):
    base_transform = [
        A.Resize(args.img_size, args.img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()
    ]

    train_transform = []
    if args.Blur:
        train_transform.append(
            A.Blur(p=args.Blur))
    if args.Blur:
        train_transform.append(
            A.ElasticTransform(p=args.Blur))

    if args.CLAHE:
        train_transform.append(A.CLAHE(clip_limit=(1, 4),
                                       tile_grid_size=(8, 8),
                                       p=args.CLAHE
                                       ))
    if args.RandomBrightnessContrast:
        train_transform.append(
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2,
                                       brightness_by_max=True,
                                       p=args.RandomBrightnessContrast
                                       ))
    if args.HueSaturationValue:
        train_transform.append(A.HueSaturationValue(hue_shift_limit=20,
                                                    sat_shift_limit=30,
                                                    val_shift_limit=20,
                                                    p=args.HueSaturationValue
                                                    ))
    if args.RGBShift:
        train_transform.append(A.RGBShift(r_shift_limit=20,
                                          g_shift_limit=20,
                                          b_shift_limit=20,
                                          p=args.RGBShift
                                          ))
    if args.RandomGamma:
        train_transform.append(A.RandomGamma(gamma_limit=(80, 120),
                                             p=args.RandomGamma
                                             ))
    if args.HorizontalFlip:
        train_transform.append(A.HorizontalFlip(p=args.HorizontalFlip))

    if args.VerticalFlip:
        train_transform.append(A.VerticalFlip(p=args.VerticalFlip))

    if args.ShiftScaleRotate:
        train_transform.append(A.ShiftScaleRotate(shift_limit=0.2,
                                                  scale_limit=0.2,
                                                  rotate_limit=10,
                                                  border_mode=args.ShiftScaleRotateMode,
                                                  p=args.ShiftScaleRotate
                                                  ))
    if args.GridDistortion:
        train_transform.append(A.GridDistortion(num_steps=5,
                                                distort_limit=(-0.3, 0.3),
                                                p=args.GridDistortion
                                                ))
    if args.MotionBlur:
        train_transform.append(A.MotionBlur(blur_limit=(3, 7),
                                            p=args.MotionBlur
                                            ))
    if args.RandomResizedCrop:
        train_transform.append(A.RandomResizedCrop(height=args.img_size,
                                                   width=args.img_size,
                                                   scale=(-0.4, 1.0),
                                                   ratio=(0.75, 1.3333333333333333),
                                                   p=args.RandomResizedCrop
                                                   ))
    if args.ImageCompression:
        train_transform.append(A.ImageCompression(quality_lower=99,
                                                  quality_upper=100,
                                                  p=args.ImageCompression
                                                  ))
    train_transform.extend(base_transform)

    train_transform = A.Compose(train_transform)
    test_transform = A.Compose(base_transform)

    return train_transform, test_transform
