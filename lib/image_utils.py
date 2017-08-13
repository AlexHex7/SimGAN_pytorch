import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import config as cfg


def normalize_img(img):
    # img_type: numpy
    img = img * 1.0 / 255
    return (img - 0.5) / 0.5


def restore_img(img):
    # img_type: numpy
    img += max(-img.min(), 0)
    if img.max() != 0:
        img /= img.max()
    img *= 255
    img = img.astype(np.uint8)
    return img


def generate_img_batch(syn_batch, ref_batch, real_batch, png_path):
    # syn_batch_type: Tensor, ref_batch_type: Tensor
    def tensor_to_numpy(img):
        img = img.numpy()
        img += max(-img.min(), 0)
        if img.max() != 0:
            img /= img.max()
        img *= 255
        img = img.astype(np.uint8)
        img = np.transpose(img, [1, 2, 0])
        return img

    syn_batch = syn_batch[:64]
    ref_batch = ref_batch[:64]
    real_batch = real_batch[:64]

    a_blank = torch.zeros(cfg.img_height, cfg.img_width*2, 1).numpy().astype(np.uint8)

    nb = syn_batch.size(0)
    # print(syn_batch.size())
    # print(ref_batch.size())
    vertical_list = []

    for index in range(0, nb, cfg.pics_line):
        st = index
        end = st + cfg.pics_line

        if end > nb:
            end = nb

        syn_line = syn_batch[st:end]
        ref_line = ref_batch[st:end]
        real_line = real_batch[st:end]
        # print('====>', nb)
        # print(syn_line.size())
        # print(ref_line.size())
        nb_per_line = syn_line.size(0)

        line_list = []

        for i in range(nb_per_line):
            #print(i, len(syn_line))
            syn_np = tensor_to_numpy(syn_line[i])
            ref_np = tensor_to_numpy(ref_line[i])
            real_np = tensor_to_numpy(real_line[i])
            a_group = np.concatenate([syn_np, ref_np, real_np], axis=1)
            line_list.append(a_group)


        fill_nb = cfg.pics_line - nb_per_line
        while fill_nb:
            line_list.append(a_blank)
            fill_nb -= 1
        # print(len(line_list))
        # print(line_list[0].shape)
        # print(line_list[1].shape)
        # print(line_list[2].shape)
        # print(line_list[3].shape)
        # print(line_list[4].shape)
        line = np.concatenate(line_list, axis=1)
        # print(line.dtype)
        vertical_list.append(line)

    imgs = np.concatenate(vertical_list, axis=0)
    if imgs.shape[-1] == 1:
        imgs = np.tile(imgs, [1, 1, 3])
    # print(imgs.shape, imgs.dtype)
    img = Image.fromarray(imgs)

    img.save(png_path, 'png')


def calc_acc(output, type='real'):
    assert type in ['real', 'refine']

    if type == 'real':
        label = Variable(torch.zeros(output.size(0)).type(torch.LongTensor)).cuda(cfg.cuda_num)
    else:
        label = Variable(torch.ones(output.size(0)).type(torch.LongTensor)).cuda(cfg.cuda_num)

    softmax_output = torch.nn.functional.softmax(output)
    acc = softmax_output.data.max(1)[1].cpu().numpy() == label.data.cpu().numpy()
    return acc.mean()
