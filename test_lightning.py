import argparse
from datareader import FusionDatasets
from torch.utils.data import DataLoader
from tools import *
from model_light import TOAFusion_lightning
import pytorch_lightning as pl



def predict(parser):

    img_vi_path, img_if_path, mask_path, text_path = prepare_path(parser.data_root, parser.mode)
    test_dataset = FusionDatasets(img_vi_path, img_if_path, text_path, mask_path, train=parser.mode,
                                  text=parser.text, parser=parser)
    test_loader = DataLoader(test_dataset,
                              batch_size=parser.batch_size,
                              shuffle=parser.shuffle,
                              pin_memory=parser.pin_memory,
                              drop_last=parser.drop_last)

    model = TOAFusion_lightning.load_from_checkpoint(parser.ckt_path)
    torch.set_grad_enabled(False)
    model.eval()
    model.set_par(parser)
    trainer = pl.Trainer(devices=parser.devices, logger=False)
    trainer.predict(model, test_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data options
    parser.add_argument('--data_root', type=str, default='/mnt/yw/Datasets/MFNet', help='Root directory of the dataset')
    parser.add_argument('--mode', type=str, default='test', help='Mode of the dataset (train/test)')

    # Testing options
    parser.add_argument('--ckt_path', type=str, default='./lightning_logs/version_0/checkpoints/last.ckpt', help='Path to the checkpoint')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--devices', type=int, default=[0], help='Device to use')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--shuffle', type=bool, default=False, help='Shuffle the data')
    parser.add_argument('--pin_memory', type=bool, default=True, help='Pin memory')
    parser.add_argument('--drop_last', type=bool, default=False, help='Drop last batch')
    parser.add_argument('--text', type=str,
                        default='We are currently working on infrared-visible image fusion and fuse all infrared information.',
                        help='Text information sent to network')
    parser.add_argument('--save_path', type=str, default='/mnt/yw/Results/TOAFusion/MFNet_new/train/original')

    args = parser.parse_args()
    predict(args)