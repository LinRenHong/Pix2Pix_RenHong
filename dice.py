import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
from math import log10

def dice(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))


def calculate_dice(fake_B, real_B):
    im_fake_B = tensor2im(fake_B)
    im_real_B = tensor2im(real_B)

    # print("fake_B shape: {}, real_B shape: {}, im_fake_B shape: {}, im_real_B shape: {}".format(fake_B.shape, real_B.shape, im_fake_B.shape, im_real_B.shape))


    real_B_dice = torch.Tensor()
    fake_B_dice = torch.Tensor()

    '''use on color'''
    im_fake_B[np.where(im_fake_B <= 100.)] = 0.
    im_fake_B[np.where(im_fake_B > 100.)] = 1.
    fake_B_dice = im_fake_B
    fake_B_dice = torch.Tensor(fake_B_dice)

    im_real_B[np.where(im_real_B <= 100)] = 0.
    im_real_B[np.where(im_real_B > 100.)] = 1.
    real_B_dice = im_real_B
    real_B_dice = torch.Tensor(real_B_dice)
    '''use on color'''

    # dice = Dice()
    return (1 - dice(fake_B_dice, real_B_dice))
    # return (dice(fake_B_dice, real_B_dice))


def calculate_PSNR(fake_img, real_img):
        criterionMSE = torch.nn.MSELoss()
        mse = criterionMSE(fake_img, real_img)
        psnr = 10 * log10(1 / mse.item())
        return psnr


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)



class Dice(_Loss):

    def __init__(self, weight=None, eps=1e-5, type="sorensen"):
        '''
        :param weight: overall weight for dice loss
        :param eps: epsilon value for smoothing
        :param type: default to "sorensen"
        '''
        self.weight = 1 if weight is None else weight
        self.eps = eps
        self.type = type

        super(Dice, self).__init__()


    def forward(self, input, labels, input_threshold=0.5, label_threshold=0.5, return_score=False, reduce=True):
        '''
        Calculate Dice Coefficient or Dice Coefficient Loss depending on the parameter return_score
        :param input: <tensor> shape-(number of examples, vector_size or image_size)
            Note: it is the output from model's softmax, tanh or sigmoid layer before binary thresholding
        :param labels: <tensor> shape-(number of examples, vector_size or image_size)
            Each vector or image from each example in labels contains 0s & 1s, which indicate the target area of interest
        :param threshold: <float>, threshold for binary thresholding
        :param return_score: return Dice Coefficient or Dice Coefficient loss (Default to False)
        :return:
            loss: Dice Coefficient loss loss if return_score == False
            score: Dice Coefficient if return_score == True
        '''

        input = input.cpu()
        labels = labels.cpu()

        # check size of the input
        if input.size(2) != labels.size(2):

            input = F.interpolate(input, labels.size(2))

        # reshape input & labels
        number_of_examples = input.size(0)
        input = input.view(number_of_examples, -1)
        labels = labels.view(number_of_examples, -1)

        # thresold the input
        thresholded_input = torch.where(input >= input_threshold,
                                        torch.ones(input.size()),
                                        torch.zeros(input.size()))

        thresholded_labels = torch.where(labels >= label_threshold,
                                         torch.ones(input.size()),
                                         torch.zeros(input.size()))

        input_op = thresholded_input if return_score else input

        overlap = thresholded_labels.mul(input_op).sum(dim=1)

        if self.type == "sorensen":

            input_sum = input_op.sum(dim=1)
            labels_sum = thresholded_labels.sum(dim=1)

        dice = ((2. * overlap + self.eps) / (input_sum + labels_sum + self.eps))

        if reduce:

            dice = dice.mean()

        if return_score:

            return dice

        else:

            return self.weight*(1.-dice)



class ModelEvaluation(object):

    @classmethod
    def calculate_dice_coefficient(cls, predictions, labels, eps=1e-5, input_threshold=0.5, label_threshold=0.5,
                                   reduce=True, verbose=0):
        '''
        Calculate dice coefficient
        :param predictions: <tensor> shape-(number of examples, vector_size or image_size)
            Note: it is the output from model's softmax, tanh or sigmoid layer before binary thresholding
        :param labels: <tensor> shape-(number of examples, vector_size or image_size)
            Each vector or image from each example in labels contains 0s & 1s, which indicate the target area of interest
        :param eps: <float>, threshold for binary thresholding
        :return:
            dice coefficient
        '''

        dice = Dice(eps=eps)

        dice_coefficient = dice(predictions, labels,
                                input_threshold=input_threshold, label_threshold=label_threshold,
                                return_score=True, reduce=reduce)

        if verbose:
            print("Dice coefficient:", dice_coefficient)

        return dice_coefficient


if __name__ == "__main__":
    # fake_img = torch.randn(1, 3, 10, 10)
    # print("This is fake image: {}".format(fake_img))
    # print("Shape is {}".format(fake_img.shape))

    # real_img = torch.randn(1, 3, 10, 10)
    # print("This is real image: {}".format(real_img))
    # print("Shape is {}".format(real_img))
    a = torch.Tensor(np.array([[[[1, 0, 0, 1], [1, 0, 1, 1]]]]))
    b = torch.Tensor(np.array([[[[1, 0, 0, 1], [1, 1, 1, 0]]]]))
    print(a.shape)
    print(b.shape)
    # testDice = dice(fake_img, real_img, return_score=True)
    # testDice = dice(a, b, return_score=True)

    # testDice = dice(a, b)
    testDice = ModelEvaluation.calculate_dice_coefficient(a, b, eps=1.)
    print(testDice)