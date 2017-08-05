import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import torch.utils.data as Data
from torchvision import transforms
from utils.image_history_buffer import ImageHistoryBuffer
import numpy as np
from PIL import Image


def generate_img_batch(syn_batch, ref_batch, png_name):
    def tensor_to_numpy(img):
        img = img.numpy()
        img += max(-img.min(), 0)
        if img.max() != 0:
            img /= img.max()
        img *= 255
        img = img.astype(np.uint8)
        img = np.transpose(img, [1, 2, 0])
        return img

    syn_batch = syn_batch[:32]
    ref_batch = ref_batch[:32]
    vertical_list = []
    # for syn, ref in zip(syn_batch, ref_batch):
    #     syn_np = tensor_to_numpy(syn)
    #     ref_np = tensor_to_numpy(ref)
    #
    #     syn_ref = np.concatenate([syn_np, ref_np], axis=1)
    #     vertical_list.append(syn_ref)
    for index in range(0, syn_batch.shape[0], 2):
        syn_np_1 = tensor_to_numpy(syn_batch[index])
        ref_np_1 = tensor_to_numpy(ref_batch[index])

        syn_np_2 = tensor_to_numpy(syn_batch[index + 1])
        ref_np_2 = tensor_to_numpy(ref_batch[index + 1])

        syn_ref = np.concatenate([syn_np_1, ref_np_1, syn_np_2, ref_np_2], axis=1)
        vertical_list.append(syn_ref)

    imgs = np.concatenate(vertical_list, axis=0)
    if imgs.shape[-1] == 1:
        imgs = np.tile(imgs, [1, 1, 3])
    img = Image.fromarray(imgs)

    img.save('res/%s' % png_name, 'png')


def calc_acc(output, type='real'):
    assert type in ['real', 'refine']

    if type == 'real':
        label = Variable(torch.zeros(output.size(0)).type(torch.LongTensor)).cuda()
    else:
        label = Variable(torch.ones(output.size(0)).type(torch.LongTensor)).cuda()

    softmax_output = torch.nn.functional.softmax(output)
    acc = softmax_output.data.max(1)[1].cpu().numpy() == label.data.cpu().numpy()
    return acc.mean()

img_width = 55
img_height = 35
img_channels = 1

# syn_path = '/home/hex/Code/data/SynthEyes_data'
# real_path = '/home/hex/Code/data/RealEyes_data'
syn_path = 'dataset/SynthEyes_data'
real_path = 'dataset/RealEyes_data'

#
# training params
#

g_pretrain = 1000
d_pretrain = 200
nb_steps = 10000000

batch_size = 128
k_d = 1  # number of discriminator updates per step
k_g = 50  # number of generative network updates per step
log_interval = 50

# refiner_model_path = 'models/r0.pkl'
refiner_model_path = None
# discriminator_model_path = 'models/d0.pkl'
discriminator_model_path = None

class ResnetBlock(nn.Module):
    def __init__(self, input_features, nb_features=64):
        super(ResnetBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(input_features, nb_features, 3, 1, 1),
            # nn.BatchNorm2d(nb_features),
            nn.ReLU(),
            nn.Conv2d(nb_features, nb_features, 3, 1, 1),
            # nn.BatchNorm2d(nb_features)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # print(x.size())
        convs = self.convs(x)
        # print(convs.size())
        sum = convs + x
        output = self.relu(sum)
        return output


class Refiner(nn.Module):
    def __init__(self, block, block_num, in_features, nb_features=64):
        super(Refiner, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_features, nb_features, 3, stride=1, padding=1),
            # nn.BatchNorm2d(nb_features)
        )

        blocks = []
        for i in range(block_num):
            blocks.append(block(nb_features, nb_features))

        self.resnet_blocks = nn.Sequential(*blocks)

        self.conv_2 = nn.Sequential(
            nn.Conv2d(nb_features, in_features, 1, 1, 0),
            nn.Tanh()
        )

    def forward(self, x):
        conv_1 = self.conv_1(x)

        res_block = self.resnet_blocks(conv_1)
        output = self.conv_2(res_block)
        return output


class Discriminator(nn.Module):
    def __init__(self, input_features):
        super(Discriminator, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(input_features, 96, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(96),

            nn.Conv2d(96, 64, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.MaxPool2d(3, 2, 1),

            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 1, 1, 0),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 2, 1, 1, 0),
            nn.ReLU(),
            # nn.BatchNorm2d(2),
            nn.MaxPool2d(3, 2, 1),
        )

    def forward(self, x):
        convs = self.convs(x)
        output = convs.view(convs.size(0), -1, 2)
        return output


img = Variable(torch.FloatTensor(10, img_channels, img_height, img_width))
r = Refiner(ResnetBlock, 4, img_channels, 64)
d = Discriminator(input_features=img_channels)
r.cuda()
d.cuda()

opt_r = torch.optim.SGD(r.parameters(), lr=0.0001)
opt_d = torch.optim.SGD(d.parameters(), lr=0.0001)
self_regularization_loss = nn.L1Loss(size_average=False)
local_adversarial_loss = nn.CrossEntropyLoss()
delta = 0.01


transform = transforms.Compose([
    transforms.ImageOps.grayscale,
    transforms.Scale((img_width, img_height)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

syn_folder = torchvision.datasets.ImageFolder(root=syn_path, transform=transform)
syn_loader = Data.DataLoader(syn_folder, batch_size=batch_size, shuffle=True)
real_folder = torchvision.datasets.ImageFolder(root=real_path, transform=transform)
real_loader = Data.DataLoader(real_folder, batch_size=batch_size, shuffle=True)


if not refiner_model_path:
    # we first train the Rθ network with just self-regularization loss for 1,000 steps
    print('pre-training the refiner network...')

    for index in range(g_pretrain):
        syn_image_batch, _ = syn_loader.__iter__().next()
        # print(synthetic_image_batch.size())
        syn_image_batch = Variable(syn_image_batch).cuda()
        r_pred = r(syn_image_batch)
        r_loss = self_regularization_loss(r_pred, syn_image_batch)
        r_loss = torch.mul(r_loss, delta)
        opt_r.zero_grad()
        r_loss.backward()
        opt_r.step()

        # log every `log_interval` steps
        if not index % log_interval:
            r.eval()
            figure_name = 'refined_image_batch_pre_train_step_{}.png'.format(index)
            print('[%d/%d] (R)reg_loss: %.4f' % (index, g_pretrain, r_loss.data[0]))
            syn_image_batch, _ = syn_loader.__iter__().next()
            # print(synthetic_image_batch)
            syn_image_batch = Variable(syn_image_batch, volatile=True).cuda()
            ref_image_batch = r(syn_image_batch)
            generate_img_batch(syn_image_batch.data.cpu(), ref_image_batch.data.cpu(), figure_name)
            r.train()

    torch.save(r.state_dict(), 'models/r0.pkl')
else:
    r.load_state_dict(torch.load('models/r0.pkl'))


if not discriminator_model_path:
    # and Dφ for 200 steps (one mini-batch for refined images, another for real)
    print('pre-training the discriminator network...')
    for index in range(d_pretrain):
        real_image_batch, _ = real_loader.__iter__().next()
        real_image_batch = Variable(real_image_batch).cuda()

        syn_image_batch, _ = syn_loader.__iter__().next()
        syn_image_batch = Variable(syn_image_batch).cuda()

        assert real_image_batch.size(0) == syn_image_batch.size(0)

        # ============ real image d ====================================================
        d_real_pred = d(real_image_batch).view(-1, 2)

        d_real_y = Variable(torch.zeros(d_real_pred.size(0)).type(torch.LongTensor)).cuda()
        d_ref_y = Variable(torch.ones(d_real_pred.size(0)).type(torch.LongTensor)).cuda()

        acc_real = calc_acc(d_real_pred, 'real')
        d_loss_real = local_adversarial_loss(d_real_pred, d_real_y)

        # ============ syn image d ====================================================
        ref_imgae_batch = r(syn_image_batch)
        d_ref_pred = d(ref_imgae_batch).view(-1, 2)

        acc_ref = calc_acc(d_ref_pred, 'refine')
        d_loss_ref = local_adversarial_loss(d_ref_pred, d_ref_y)

        d_loss = d_loss_real + d_loss_ref
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        print('[%d/%d] (D)d_loss:%f  acc_real:%.4f acc_ref:%.4f'
              % (index, d_pretrain, d_loss.data[0], acc_real.mean(), acc_ref.mean()))
    torch.save(d.state_dict(), 'models/d0.pkl')
else:
    d.load_state_dict(torch.load('models/d0.pkl'))


image_history_buffer = ImageHistoryBuffer((0, img_channels, img_height, img_width), batch_size * 100, batch_size)

for epoch in range(nb_steps):
    print('Step: {} of {}.'.format(epoch, nb_steps))

    # train the R =================================================
    total_r_loss = 0.0
    total_r_loss_reg_scale = 0.0
    total_r_loss_adv = 0.0
    total_acc_adv = 0.0
    for index in range(k_g):
        syn_image_batch, _ = syn_loader.__iter__().next()
        syn_image_batch = Variable(syn_image_batch).cuda()

        ref_image_batch = r(syn_image_batch)

        d_ref_pred = d(ref_image_batch).view(-1, 2)
        d_real_y = Variable(torch.zeros(d_ref_pred.size(0)).type(torch.LongTensor)).cuda()

        acc_adv = calc_acc(d_ref_pred, 'real')

        r_loss_reg = self_regularization_loss(ref_image_batch, syn_image_batch)
        r_loss_reg_scale = torch.mul(r_loss_reg, delta)
        r_loss_adv = local_adversarial_loss(d_ref_pred, d_real_y)

        r_loss = r_loss_reg_scale + r_loss_adv

        opt_r.zero_grad()
        opt_d.zero_grad()
        # loss_1.backward()
        # loss_2.backward()
        r_loss.backward()
        opt_r.step()

        total_r_loss += r_loss
        total_r_loss_reg_scale += r_loss_reg_scale
        total_r_loss_adv += r_loss_adv
        total_acc_adv += acc_adv
    mean_r_loss = total_r_loss / k_g
    mean_r_loss_reg_scale = total_r_loss_reg_scale / k_g
    mean_r_loss_adv = total_r_loss_adv / k_g
    mean_acc_adv = total_acc_adv / k_g

    print('(R)r_loss:%f r_loss_reg:%f, r_loss_adv:%f acc_ref_to_real:%.4f'
            % (mean_r_loss.data[0], mean_r_loss_reg_scale.data[0], mean_r_loss_adv.data[0], mean_acc_adv))

    # train the D =================================================
    for index in range(k_d):
        real_image_batch, _ = real_loader.__iter__().next()
        syn_image_batch, _ = syn_loader.__iter__().next()
        assert real_image_batch.size(0) == syn_image_batch.size(0)

        real_image_batch = Variable(real_image_batch).cuda()
        syn_image_batch = Variable(syn_image_batch).cuda()

        ref_image_batch = r(syn_image_batch)

        # use a history of refined images
        half_batch_from_image_history = image_history_buffer.get_from_image_history_buffer()

        image_history_buffer.add_to_image_history_buffer(ref_image_batch.cpu().data.numpy())

        if len(half_batch_from_image_history):
            torch_type = torch.from_numpy(half_batch_from_image_history)
            v_type = Variable(torch_type).cuda()
            ref_image_batch[:batch_size // 2] = v_type

        d_real_pred = d(real_image_batch).view(-1, 2)
        d_real_y = Variable(torch.zeros(d_real_pred.size(0)).type(torch.LongTensor)).cuda()
        d_loss_real = local_adversarial_loss(d_real_pred, d_real_y)

        acc_real = calc_acc(d_real_pred, 'real')

        d_ref_pred = d(ref_image_batch).view(-1, 2)
        d_ref_y = Variable(torch.ones(d_ref_pred.size(0)).type(torch.LongTensor)).cuda()
        d_loss_ref = local_adversarial_loss(d_ref_pred, d_ref_y)

        acc_ref = calc_acc(d_ref_pred, 'refine')

        d_loss = d_loss_real + d_loss_ref
        d.zero_grad()
        d_loss.backward()
        opt_d.step()

        print('(D)d_loss:%f real_loss:%f refine_loss:%f acc_real:%.4f acc_ref:%.4f'
              % (d_loss.data[0], d_loss_real.data[0], d_loss_ref.data[0], acc_real, acc_ref))

    if not epoch % log_interval:
        r.eval()
        synthetic_image_batch, _ = syn_loader.__iter__().next()
        synthetic_image_batch = Variable(synthetic_image_batch, volatile=True).cuda()
        ref_output = r(synthetic_image_batch)
        generate_img_batch(synthetic_image_batch.cpu().data, ref_output.cpu().data, '%d.png' % epoch)
        r.train()
        print('Save two model dict.')
        torch.save(d.state_dict(), 'models/d.pkl')
        torch.save(r.state_dict(), 'models/r.pkl')
