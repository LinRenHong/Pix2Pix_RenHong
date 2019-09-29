import argparse

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import datetime

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import *
from datasets import  *
from dice import ModelEvaluation
from dice import calculate_PSNR


# def sample_images():
#     """Saves a generated sample from the validation set"""
#     imgs = next(iter(val_dataloader))
#     real_A = Variable(imgs["A"].type(Tensor)) # change
#     real_B = Variable(imgs["B"].type(Tensor)) # change
#
#     # real_A = Variable(imgs["B"].type(Tensor))
#     # real_B = Variable(imgs["A"].type(Tensor))
#     fake_B = generator(real_A)
#
#     ### calculate val dice ###
#     total_val_dice = 0.0
#     for i in range(0, opt.num_display):
#         val_dice = calculate_dice(fake_B[i].unsqueeze_(0), real_B[i].unsqueeze_(0))
#         # total_val_dice = total_val_dice + val_dice
#         total_val_dice = total_val_dice + val_dice
#
#     avg_val_dice = total_val_dice / opt.num_display
#     print("Avg dice: {}".format(avg_val_dice))
#     ### calculate val dice ###
#
#     img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
#     save_image(img_sample, "images/%s_test.png" % (opt.dataset_name), nrow=5, normalize=True)
#
#
# def single_sample_images():
#     """Saves a generated sample from the validation set"""
#     imgs = next(iter(val_dataloader))
#     real_A = Variable(imgs.type(Tensor))
#
#     fake_B = generator(real_A)
#     img_sample = torch.cat((real_A.data, fake_B.data), -2)
#     save_image(img_sample, "images/%s_test.png" % (opt.dataset_name), nrow=5, normalize=True)

def sample_and_val_images_for_RGB():

    total_val_dice = 0.0
    total_val_psnr = 0.0

    temp_real_A = 0
    temp_real_B = 0
    temp_fake_B = 0

    sample_num = 9
    want_sample = random.sample(range(1, len(val_dataset)), sample_num)

    for i, batch in enumerate(val_dataloader):

        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        fake_B = generator(real_A)


        # val_dice = calculate_dice(fake_B, real_B)
        val_dice = ModelEvaluation.calculate_dice_coefficient(fake_B, real_B, input_threshold=0.0, label_threshold=0.0, eps=1.)
        total_val_dice = total_val_dice + val_dice

        val_psnr = calculate_PSNR(fake_B, real_B)
        total_val_psnr = total_val_psnr + val_psnr

        if i == 0 or i in want_sample:
            if i == 0:
                temp_real_A = real_A
                temp_real_B = real_B
                temp_fake_B = fake_B
            else:
                temp_real_A = torch.cat((temp_real_A, real_A), dim=0)
                temp_real_B = torch.cat((temp_real_B, real_B), dim=0)
                temp_fake_B = torch.cat((temp_fake_B, fake_B), dim=0)


    # avg_val_dice = total_val_dice / opt.val_batch_size
    avg_val_dice = total_val_dice / len(val_dataset)
    avg_val_psnr = total_val_psnr / len(val_dataset)

    print("Dice: {}".format(avg_val_dice))
    print("PSNR: {}".format(avg_val_psnr))

    '''------------------------------val_dice and psnr----------------------------'''

    img_sample = torch.cat((temp_real_A.data, temp_fake_B.data, temp_real_B.data), -2)
    save_image(img_sample, "images/%s_test(%s)-val_index(%d).png" % (today, dataset_name, val_idx), nrow=5, normalize=True)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    # parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
    # parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    # parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    # parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    # parser.add_argument("--num_display", type=int, default=10, help="number of image display")
    # parser.add_argument("--load_model_path", type=str, default='best_model/wound/best_generator_581.pth', help="model path")
    # opt = parser.parse_args()
    # print(opt)
    #
    # # Configure dataloaders
    # transforms_ =   [
    #                     transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    #                     transforms.ToTensor(),
    #                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                 ]
    #
    # ### When use combine image ###
    # val_dataloader = DataLoader(
    #                                 ImageDataset("dataset/%s" % opt.dataset_name, transforms_=transforms_, mode="val"),
    #                                 batch_size=opt.num_display,
    #                                 shuffle=True,
    #                                 num_workers=1,
    #                             )
    #
    # # ### Whne use single image ###
    # # val_dataloader = DataLoader(
    # #     SingleImageDataset("dataset/%s" % opt.dataset_name, transforms_=transforms_, mode="test"),
    # #     batch_size=opt.num_display,
    # #     shuffle=True,
    # #     num_workers=1,
    # # )
    #
    # cuda = True if torch.cuda.is_available() else False
    #
    # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    #
    # generator = GeneratorUNet()
    #
    # if cuda:
    #     generator = generator.cuda()
    #
    # # generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
    # generator.load_state_dict(torch.load(opt.load_model_path))
    #
    # sample_images()
    # # single_sample_images()

    opt = config
    print(opt)

    today = datetime.date.today()
    today = "%d%02d%02d" % (today.year, today.month, today.day)
    dataset_name = os.path.split(opt.dataset_path)[-1]
    val_idx = 5

    # Configure dataloaders
    transforms_train = [
        transforms.Resize((opt.img_crop_height, opt.img_crop_width), Image.BICUBIC),
        transforms.ToTensor(),
    ]

    train_csv_file = opt.dataset_path + "/" + "list.csv"
    img_container = opt.dataset_path

    val_dataset = ReadCsvImageDataSet(csv_file_path_=train_csv_file,
                                      transforms_=transforms_train,
                                      image_container_=img_container,
                                      mode='train',
                                      validation_index_=val_idx)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    cuda = True if torch.cuda.is_available() else False

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    generator = GeneratorUNet()

    if cuda:
        generator = generator.cuda()

    generator.load_state_dict(torch.load(opt.load_model_path))
    sample_and_val_images_for_RGB()