import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

'''---------------------------------------------add new code --------------------------------------------'''
from tensorboardX import SummaryWriter
from dice import calculate_dice
from dice import calculate_PSNR
'''---------------------------------------------add new code --------------------------------------------'''

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs["A"].type(Tensor)) # change
    real_B = Variable(imgs["B"].type(Tensor)) # change

    # real_A = Variable(imgs["B"].type(Tensor)) # origin
    # real_B = Variable(imgs["A"].type(Tensor)) # origin

    fake_B = generator(real_A)

    '''------------------------------val_dice and psnr----------------------------'''
    total_val_dice = 0.0
    total_val_psnr = 0.0
    for i in range(0, opt.val_batch_size):
        val_dice = calculate_dice(fake_B[i].unsqueeze_(0), real_B[i].unsqueeze_(0))
        total_val_dice = total_val_dice + val_dice

        val_psnr = calculate_PSNR(fake_B[i].unsqueeze_(0), real_B[i].unsqueeze_(0))
        total_val_psnr = total_val_psnr + val_psnr


    avg_val_dice = total_val_dice / opt.val_batch_size
    avg_val_psnr = total_val_psnr / opt.val_batch_size

    writer.add_scalar(tag='Dice_val', scalar_value=avg_val_dice.item(), global_step=epoch)
    writer.add_scalar(tag='PSNR_val', scalar_value=avg_val_psnr, global_step=epoch)
    '''------------------------------val_dice and psnr----------------------------'''

    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    '''---------------------------------------------add new code --------------------------------------------'''
    parser.add_argument("--val_batch_size", type=int, default=10, help="size of the validation batches")
    '''---------------------------------------------add new code --------------------------------------------'''
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument(
        "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
    )
    # parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints") # origin
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between model checkpoints")
    opt = parser.parse_args()
    print(opt)


    '''---------------------------------------------add new code --------------------------------------------'''
    tb_log_name = os.path.join("tf_log", "tf_" + "wound_change_with_data_augmentation") # have to change
    writer = SummaryWriter(tb_log_name)
    '''---------------------------------------------add new code --------------------------------------------'''

    os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
    os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()

    # Loss weight of L1 pixel-wise loss between translated image and real image
    lambda_pixel = 100

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

    # Initialize generator and discriminator
    generator = GeneratorUNet()
    discriminator = Discriminator()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
        discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Configure dataloaders
    transforms_ = [
        transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
        ### data agumentation ###
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(90),
        # transforms.RandomCrop(256),
        ### data augmentation ###
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    # dataloader = DataLoader(
    #     DataAugmentationImageDataset("dataset/%s" % opt.dataset_name, transforms_=transforms_),
    #     batch_size=opt.batch_size,
    #     shuffle=True,
    #     num_workers=opt.n_cpu,
    # )

    '''---------------------------------------------add new dataset--------------------------------------------'''
    ### Trainning dataset read from CSV ###
    train_csv_file = "csv_and_image/list.csv"
    img_container = 'csv_and_image'

    train_csv_file = "dataset/wound_20190620/list.csv"
    img_container = 'dataset/wound_20190620'

    train_dataset = ReadCsvImageDataSet(csv_file_path_=train_csv_file,
                                        transforms_=transforms_,
                                        image_container_=img_container,
                                        mode='train',
                                        validation_index_=5)

    dataloader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )
    '''---------------------------------------------add new dataset--------------------------------------------'''
    # val_dataloader = DataLoader(
    #     ImageDataset("dataset/%s" % opt.dataset_name, transforms_=transforms_, mode="val"),
    #     batch_size=opt.val_batch_size,
    #     shuffle=True,
    #     num_workers=1,
    # )

    '''---------------------------------------------add new dataset--------------------------------------------'''
    # ### Validation dataset read from CSV ###
    val_dataset = ReadCsvImageDataSet(csv_file_path_=train_csv_file,
                                        transforms_=transforms_,
                                        image_container_=img_container,
                                        mode='val',
                                        validation_index_=5)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=opt.val_batch_size,
        shuffle=True,
        num_workers=1,
    )
    '''---------------------------------------------add new dataset--------------------------------------------'''
    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    # ----------
    #  Training
    # ----------

    prev_time = time.time()

    '''---------------------------------------------add new code---------------------------------------------'''
    best_train_dice = 0.0
    '''---------------------------------------------add new code---------------------------------------------'''

    for epoch in range(opt.epoch, opt.n_epochs):
        '''---------------------------------------------add new code---------------------------------------------'''
        total_train_dice = 0.0
        total_train_psnr = 0.0
        '''---------------------------------------------add new code---------------------------------------------'''
        for i, batch in enumerate(dataloader):

            # Model inputs
            real_A = Variable(batch["A"].type(Tensor)) # change
            real_B = Variable(batch["B"].type(Tensor)) # change
            # real_A = Variable(imgs["B"].type(Tensor))
            # real_B = Variable(imgs["A"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # GAN loss
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, valid)
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)
            '''---------------------------------------------add new code --------------------------------------------'''
            ### Dice ###
            train_dice = calculate_dice(fake_B, real_B)
            total_train_dice = total_train_dice + train_dice

            ### PSNR ###
            train_psnr = calculate_PSNR(fake_B, real_B)
            total_train_psnr = total_train_psnr + train_psnr

            # dice = Variable(loss_dice, requires_grad=True)
            # dice.backward()
            '''---------------------------------------------add new code --------------------------------------------'''
            # Total loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel

            loss_G.backward()

            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f, dice: %f, PSNR: %.4f dB] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_pixel.item(),
                    loss_GAN.item(),
                    train_dice.item(),
                    train_psnr,
                    time_left,
                )
            )



            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)
        '''-----------------------------------------------add new code------------------------------------------------'''
        ### TensorBoard ###
        writer.add_scalar(tag='Loss_D', scalar_value=loss_D.item(), global_step=epoch)
        writer.add_scalar(tag='Loss_G', scalar_value=loss_G.item(), global_step=epoch)
        writer.add_scalar(tag='Loss_pixel', scalar_value=loss_pixel.item(), global_step=epoch)
        writer.add_scalar(tag='Loss_GAN', scalar_value=loss_GAN.item(), global_step=epoch)
        # writer.add_scalar(tag='Dice', scalar_value=train_dice.item(), global_step=epoch)
        avg_train_dice = total_train_dice / len(dataloader)
        writer.add_scalar(tag='Dice_train', scalar_value=avg_train_dice.item(), global_step=epoch)
        writer.add_scalars(main_tag='Loss_D/Loss_G', tag_scalar_dict={'Loss_D': loss_D.item(),
                                                                      'loss_G': loss_G.item()}, global_step=epoch )
        avg_train_psnr = total_train_psnr / len(dataloader)
        writer.add_scalar(tag='PSNR_train', scalar_value=avg_train_psnr, global_step=epoch)
        '''-----------------------------------------------add new code------------------------------------------------'''
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
            torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))

        '''-----------------------------------------------add new code------------------------------------------------'''
        ### Save best  model ###
        if avg_train_dice > best_train_dice:
            best_train_dice = avg_train_dice
            torch.save(generator.state_dict(), "saved_models/%s/best_generator_%d.pth" % (opt.dataset_name, epoch))
            torch.save(discriminator.state_dict(), "saved_models/%s/best_discriminator_%d.pth" % (opt.dataset_name, epoch))
        '''-----------------------------------------------add new code------------------------------------------------'''


    '''-----------------------------------------------add new code------------------------------------------------'''
    ### Save latest model ###
    torch.save(generator.state_dict(), "saved_models/%s/generator_%s.pth" % (opt.dataset_name, opt.n_epochs))
    torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%s.pth" % (opt.dataset_name, opt.n_epochs))
    '''-----------------------------------------------add new code------------------------------------------------'''