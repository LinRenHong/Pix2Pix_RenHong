""" config.py
"""
import argparse

parser = argparse.ArgumentParser("Pix2Pix")
parser.add_argument("--load_model_path", type=str, default=r'saved_models\#C005-20190813-(Dataset_0812)-(Pix2Pix)-ep(600)-bs(1)-lr(0.00002)-img_size(286, 286, 3)-crop_size(256, 256, 3)-val_index(5)/best_generator_595.pth', help="model path")

parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("--dataset_path", type=str, default=r"dataset/csv_and_image", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
'''---------------------------------------------add new code --------------------------------------------'''
parser.add_argument("--val_batch_size", type=int, default=1, help="size of the validation batches")
parser.add_argument("--val_index", type=int, default=1, help="index of the validation dataset")
'''---------------------------------------------add new code --------------------------------------------'''
parser.add_argument("--lr", type=float, default=2e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

# image size
parser.add_argument("--img_height", type=int, default=286, help="size of image height")
parser.add_argument("--img_width", type=int, default=286, help="size of image width")
parser.add_argument("--img_crop_height", type=int, default=256, help="size of image height after crop")
parser.add_argument("--img_crop_width", type=int, default=256, help="size of image width after crop")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")

parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
# parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints") # origin
parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")

## parse and save config.
config = parser.parse_args()
# config, _ = parser.parse_known_args()
