import argparse

from rfdetr import RFDETRMedium


def train(config):
    model = RFDETRMedium()
    model.train(**config)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune RF-DETR medium model.")
    parser.add_argument("--data-dir", type=str, help="Directory containing dataset", default='./dataset/')
    parser.add_argument("--epochs", type=int, help="Num of epochs", default=10)
    parser.add_argument("--batch-size", type=int, help="Batch size", default=12)
    parser.add_argument("--lr", type=float, help="Batch size", default=1.5e-4)
    parser.add_argument("--output-dir", type=str, help="Model wights output dir", default='./output/epoch-test')
    parser.add_argument("--use-wandb", type=bool, help="Model wights output dir", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = dict(
        dataset_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=1,
        lr=args.lr,
        output_dir=args.output_dir,
        dataset_file="roboflow",  # 'coco', 'roboflow'
        # coco_path='./dataset',
        num_classes=7,
        lr_scheduler='cosine',
        warmup_epochs=1,
        run_test=True,
        seec=0
    )
    if args.use_wandb:
        wandb_config = dict(
            wandb=True,
            project='RF-DETR-lr',
            run='Test-Roboflow',
        )
        config.update(wandb_config)

    train(config)
