from datareader import FusionDatasets
from torch.utils.data import DataLoader
from tools import *
from model_light import TOAFusion_lightning
import pytorch_lightning as pl
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint


def train(parser):
    set_seed(parser.seeds)
    img_vi_path, img_if_path, mask_path, text_path = prepare_path(parser.data_root, parser.mode)
    train_dataset = FusionDatasets(img_vi_path, img_if_path, text_path, mask_path, train=parser.mode, parser=parser)
    train_loader = DataLoader(train_dataset,
                              batch_size=parser.batch_size,
                              shuffle=parser.shuffle,
                              num_workers=args.num_workers,
                              pin_memory=parser.pin_memory,
                              drop_last=parser.drop_last)
    if parser.train_FMB:
        model = TOAFusion_lightning(parser, text_path)
        checkpoint = torch.load(parser.MFNet_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])

        print("Training Second Stage")
    else:
        model = TOAFusion_lightning(parser, text_path)
        print("Training From Begining")

    checkpoint_callback = ModelCheckpoint(
        filename='checkpoints-{epoch}',
        every_n_epochs=parser.every_n_epochs,
        save_top_k=-1,
        save_last=True
    )
    trainer = pl.Trainer(max_epochs=parser.epochs, devices=parser.devices,
                          log_every_n_steps=1, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data options
    parser.add_argument('--data_root', type=str, default='/mnt/yw/Datasets/MFNet', help='Root directory of the dataset')
    parser.add_argument('--mode', type=str, default='train', help='Mode of the dataset (train/test)')

    # Training options
    parser.add_argument('--seeds', type=int, default=427, help='Set seeds')
    parser.add_argument('--epochs', type=int, default=700, help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=15, help='Number of workers')
    parser.add_argument('--devices', type=int, default=[0,1], help='Device to use')
    parser.add_argument('--lr', type=float, default=4e-4, help='Learning rate')
    parser.add_argument('--lr_end', type=float, default=4e-6, help='End learning rate')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle the data')
    parser.add_argument('--pin_memory', type=bool, default=True, help='Pin memory')
    parser.add_argument('--drop_last', type=bool, default=True, help='Drop last batch')
    parser.add_argument('--text_type', type=str, default=['people', 'car','original','all'],
                        help='Text type of the dataset')
    parser.add_argument('--resize', type=bool, default=False, help='Traning data is resized')
    parser.add_argument('--every_n_epochs', type=int, default=100, help='Save checkpoint every n epochs')
    parser.add_argument('--weight_path', type=str, default = './chekpoints')

    # Resume options
    parser.add_argument('--resume', type=bool, default=False, help='Training from the checkpoint')
    parser.add_argument('--ckt_path', type=str, default=None, help='Lastly training checkpoint path')
    parser.add_argument('--train_FMB', type=bool, default=False)
    parser.add_argument('--MFNet_path', type=str, default='./lightning_logs/MFNet/checkpoints/last.ckpt')

    # Loss options
    parser.add_argument('--ratio_int', type=float, default=1.3, help='Ratio of intensity loss')
    parser.add_argument('--ratio_grad', type=float, default=0.8, help='Ratio of gradient loss')
    parser.add_argument('--ratio_mask', type=float, default=1.2, help='Ratio of mask loss')

    args = parser.parse_args()
    train(args)
