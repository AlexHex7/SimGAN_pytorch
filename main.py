import torch
import torch.utils.data as Data
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from lib.image_history_buffer import ImageHistoryBuffer
from lib.network import Discriminator, Refiner
from lib.image_utils import generate_img_batch, calc_acc
import config as cfg
import os


class Main(object):
    def __init__(self):
        # network
        self.R = None
        self.D = None
        self.opt_R = None
        self.opt_D = None
        self.self_regularization_loss = None
        self.local_adversarial_loss = None
        self.delta = None

        # data
        self.syn_train_loader = None
        self.real_loader = None

    def build_network(self):
        print('=' * 50)
        print('Building network...')
        self.R = Refiner(4, cfg.img_channels, nb_features=64)
        self.D = Discriminator(input_features=cfg.img_channels)

        if cfg.cuda_use:
            self.R.cuda(cfg.cuda_num)
            self.D.cuda(cfg.cuda_num)

        self.opt_R = torch.optim.Adam(self.R.parameters(), lr=cfg.r_lr)
        self.opt_D = torch.optim.SGD(self.D.parameters(), lr=cfg.d_lr)
        self.self_regularization_loss = nn.L1Loss(size_average=False)
        self.local_adversarial_loss = nn.CrossEntropyLoss(size_average=True)
        self.delta = cfg.delta

    def load_data(self):
        print('=' * 50)
        print('Loading data...')
        transform = transforms.Compose([
            transforms.ImageOps.grayscale,
            transforms.Scale((cfg.img_width, cfg.img_height)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        syn_train_folder = torchvision.datasets.ImageFolder(root=cfg.syn_path, transform=transform)
        # print(syn_train_folder)
        self.syn_train_loader = Data.DataLoader(syn_train_folder, batch_size=cfg.batch_size, shuffle=True,
                                                pin_memory=True)
        print('syn_train_batch %d' % len(self.syn_train_loader))

        real_folder = torchvision.datasets.ImageFolder(root=cfg.real_path, transform=transform)
        # real_folder.imgs = real_folder.imgs[:2000]
        self.real_loader = Data.DataLoader(real_folder, batch_size=cfg.batch_size, shuffle=True,
                                           pin_memory=True)
        print('real_batch %d' % len(self.real_loader))

    def pre_train_r(self):
        print('=' * 50)
        if cfg.ref_pre_path:
            print('Loading R_pre from %s' % cfg.ref_pre_path)
            self.R.load_state_dict(torch.load(cfg.ref_pre_path))
            return

        # we first train the Rθ network with just self-regularization loss for 1,000 steps
        print('pre-training the refiner network %d times...' % cfg.r_pretrain)

        for index in range(cfg.r_pretrain):
            syn_image_batch, _ = self.syn_train_loader.__iter__().next()
            syn_image_batch = Variable(syn_image_batch).cuda(cfg.cuda_num)

            self.R.train()
            ref_image_batch = self.R(syn_image_batch)

            r_loss = self.self_regularization_loss(ref_image_batch, syn_image_batch)
            # r_loss = torch.div(r_loss, cfg.batch_size)
            r_loss = torch.mul(r_loss, self.delta)

            self.opt_R.zero_grad()
            r_loss.backward()
            self.opt_R.step()

            # log every `log_interval` steps
            if (index % cfg.r_pre_per == 0) or (index == cfg.r_pretrain - 1):
                # figure_name = 'refined_image_batch_pre_train_step_{}.png'.format(index)
                print('[%d/%d] (R)reg_loss: %.4f' % (index, cfg.r_pretrain, r_loss.data[0]))

                syn_image_batch, _ = self.syn_train_loader.__iter__().next()
                syn_image_batch = Variable(syn_image_batch, volatile=True).cuda(cfg.cuda_num)

                real_image_batch, _ = self.real_loader.__iter__().next()
                real_image_batch = Variable(real_image_batch, volatile=True)

                self.R.eval()
                ref_image_batch = self.R(syn_image_batch)

                figure_path = os.path.join(cfg.train_res_path, 'refined_image_batch_pre_train_%d.png' % index)
                generate_img_batch(syn_image_batch.data.cpu(), ref_image_batch.data.cpu(),
                                   real_image_batch.data, figure_path)
                self.R.train()

                print('Save R_pre to models/R_pre.pkl')
                torch.save(self.R.state_dict(), 'models/R_pre.pkl')

    def pre_train_d(self):
        print('=' * 50)
        if cfg.disc_pre_path:
            print('Loading D_pre from %s' % cfg.disc_pre_path)
            self.D.load_state_dict(torch.load(cfg.disc_pre_path))
            return

        # and Dφ for 200 steps (one mini-batch for refined images, another for real)
        print('pre-training the discriminator network %d times...' % cfg.r_pretrain)

        self.D.train()
        self.R.eval()
        for index in range(cfg.d_pretrain):
            real_image_batch, _ = self.real_loader.__iter__().next()
            real_image_batch = Variable(real_image_batch).cuda(cfg.cuda_num)

            syn_image_batch, _ = self.syn_train_loader.__iter__().next()
            syn_image_batch = Variable(syn_image_batch).cuda(cfg.cuda_num)

            assert real_image_batch.size(0) == syn_image_batch.size(0)

            # ============ real image D ====================================================
            # self.D.train()
            d_real_pred = self.D(real_image_batch).view(-1, 2)

            d_real_y = Variable(torch.zeros(d_real_pred.size(0)).type(torch.LongTensor)).cuda(cfg.cuda_num)
            d_ref_y = Variable(torch.ones(d_real_pred.size(0)).type(torch.LongTensor)).cuda(cfg.cuda_num)

            acc_real = calc_acc(d_real_pred, 'real')
            d_loss_real = self.local_adversarial_loss(d_real_pred, d_real_y)
            # d_loss_real = torch.div(d_loss_real, cfg.batch_size)

            # ============ syn image D ====================================================
            # self.R.eval()
            ref_image_batch = self.R(syn_image_batch)

            # self.D.train()
            d_ref_pred = self.D(ref_image_batch).view(-1, 2)

            acc_ref = calc_acc(d_ref_pred, 'refine')
            d_loss_ref = self.local_adversarial_loss(d_ref_pred, d_ref_y)
            # d_loss_ref = torch.div(d_loss_ref, cfg.batch_size)

            d_loss = d_loss_real + d_loss_ref
            self.opt_D.zero_grad()
            d_loss.backward()
            self.opt_D.step()

            if (index % cfg.d_pre_per == 0) or (index == cfg.d_pretrain - 1):
                print('[%d/%d] (D)d_loss:%f  acc_real:%.2f%% acc_ref:%.2f%%'
                      % (index, cfg.d_pretrain, d_loss.data[0], acc_real, acc_ref))

        print('Save D_pre to models/D_pre.pkl')
        torch.save(self.D.state_dict(), 'models/D_pre.pkl')

    def train(self):
        print('=' * 50)
        print('Training...')
        image_history_buffer = ImageHistoryBuffer((0, cfg.img_channels, cfg.img_height, cfg.img_width),
                                                  cfg.buffer_size * 10, cfg.batch_size)
        for step in range(cfg.train_steps):
            print('Step[%d/%d]' % (step, cfg.train_steps))

            # ========= train the R =========
            self.D.eval()
            self.R.train()

            for p in self.D.parameters():
                p.requires_grad = False

            total_r_loss = 0.0
            total_r_loss_reg_scale = 0.0
            total_r_loss_adv = 0.0
            total_acc_adv = 0.0

            for index in range(cfg.k_r):
                syn_image_batch, _ = self.syn_train_loader.__iter__().next()
                syn_image_batch = Variable(syn_image_batch).cuda(cfg.cuda_num)

                ref_image_batch = self.R(syn_image_batch)
                d_ref_pred = self.D(ref_image_batch).view(-1, 2)

                d_real_y = Variable(torch.zeros(d_ref_pred.size(0)).type(torch.LongTensor)).cuda(cfg.cuda_num)

                acc_adv = calc_acc(d_ref_pred, 'real')

                r_loss_reg = self.self_regularization_loss(ref_image_batch, syn_image_batch)
                r_loss_reg_scale = torch.mul(r_loss_reg, self.delta)
                # r_loss_reg_scale = torch.div(r_loss_reg_scale, cfg.batch_size)

                r_loss_adv = self.local_adversarial_loss(d_ref_pred, d_real_y)
                # r_loss_adv = torch.div(r_loss_adv, cfg.batch_size)

                r_loss = r_loss_reg_scale + r_loss_adv

                self.opt_R.zero_grad()
                self.opt_D.zero_grad()
                r_loss.backward()
                self.opt_R.step()

                total_r_loss += r_loss
                total_r_loss_reg_scale += r_loss_reg_scale
                total_r_loss_adv += r_loss_adv
                total_acc_adv += acc_adv
            mean_r_loss = total_r_loss / cfg.k_r
            mean_r_loss_reg_scale = total_r_loss_reg_scale / cfg.k_r
            mean_r_loss_adv = total_r_loss_adv / cfg.k_r
            mean_acc_adv = total_acc_adv / cfg.k_r

            print('(R)r_loss:%.4f r_loss_reg:%.4f, r_loss_adv:%f(%.2f%%)'
                  % (mean_r_loss.data[0], mean_r_loss_reg_scale.data[0], mean_r_loss_adv.data[0], mean_acc_adv))

            # ========= train the D =========
            self.R.eval()
            self.D.train()
            for p in self.D.parameters():
                p.requires_grad = True

            for index in range(cfg.k_d):
                real_image_batch, _ = self.real_loader.__iter__().next()
                syn_image_batch, _ = self.syn_train_loader.__iter__().next()
                assert real_image_batch.size(0) == syn_image_batch.size(0)

                real_image_batch = Variable(real_image_batch).cuda(cfg.cuda_num)
                syn_image_batch = Variable(syn_image_batch).cuda(cfg.cuda_num)

                ref_image_batch = self.R(syn_image_batch)

                # use a history of refined images
                half_batch_from_image_history = image_history_buffer.get_from_image_history_buffer()
                image_history_buffer.add_to_image_history_buffer(ref_image_batch.cpu().data.numpy())

                if len(half_batch_from_image_history):
                    torch_type = torch.from_numpy(half_batch_from_image_history)
                    v_type = Variable(torch_type).cuda(cfg.cuda_num)
                    ref_image_batch[:cfg.batch_size // 2] = v_type

                d_real_pred = self.D(real_image_batch).view(-1, 2)
                d_real_y = Variable(torch.zeros(d_real_pred.size(0)).type(torch.LongTensor)).cuda(cfg.cuda_num)
                d_loss_real = self.local_adversarial_loss(d_real_pred, d_real_y)
                # d_loss_real = torch.div(d_loss_real, cfg.batch_size)
                acc_real = calc_acc(d_real_pred, 'real')

                d_ref_pred = self.D(ref_image_batch).view(-1, 2)
                d_ref_y = Variable(torch.ones(d_ref_pred.size(0)).type(torch.LongTensor)).cuda(cfg.cuda_num)
                d_loss_ref = self.local_adversarial_loss(d_ref_pred, d_ref_y)
                # d_loss_ref = torch.div(d_loss_ref, cfg.batch_size)
                acc_ref = calc_acc(d_ref_pred, 'refine')

                d_loss = d_loss_real + d_loss_ref

                self.D.zero_grad()
                d_loss.backward()
                self.opt_D.step()

                print('(D)d_loss:%.4f real_loss:%.4f(%.2f%%) refine_loss:%.4f(%.2f%%)'
                      % (d_loss.data[0] / 2, d_loss_real.data[0], acc_real, d_loss_ref.data[0], acc_ref))

            if step % cfg.save_per == 0:
                print('Save two model dict.')
                torch.save(self.D.state_dict(), cfg.D_path % step)
                torch.save(self.R.state_dict(), cfg.R_path % step)

                real_image_batch, _ = self.real_loader.__iter__().next()
                syn_image_batch, _ = self.syn_train_loader.__iter__().next()
                real_image_batch = Variable(real_image_batch, volatile=True).cuda(cfg.cuda_num)
                syn_image_batch = Variable(syn_image_batch, volatile=True).cuda(cfg.cuda_num)

                self.R.eval()
                ref_image_batch = self.R(syn_image_batch)
                self.generate_batch_train_image(syn_image_batch, ref_image_batch, real_image_batch, step_index=step)

    def generate_batch_train_image(self, syn_image_batch, ref_image_batch, real_image_batch, step_index=-1):
        print('=' * 50)
        print('Generating a batch of training images...')
        self.R.eval()

        pic_path = os.path.join(cfg.train_res_path, 'step_%d.png' % step_index)
        generate_img_batch(syn_image_batch.cpu().data, ref_image_batch.cpu().data, real_image_batch.cpu().data, pic_path)
        print('=' * 50)

    # def generate_all_train_image(self):
    #     print('=' * 50)
    #     print('Generating all training images...')
    #     self.R.eval()
    #
    #     for index, (syn_image_batch, _) in enumerate(self.syn_train_loader):
    #         pic_path = os.path.join(cfg.final_res_path, 'batch_%d.png' % index)
    #
    #         syn_image_batch = Variable(syn_image_batch, volatile=True).cuda(cfg.cuda_num)
    #         ref_image_batch = self.R(syn_image_batch)
    #         generate_img_batch(syn_image_batch.cpu().data, ref_image_batch.cpu().data, pic_path)
    #     print('=' * 50)


if __name__ == '__main__':
    obj = Main()
    obj.build_network()

    obj.load_data()

    obj.pre_train_r()
    obj.pre_train_d()
    obj.train()

    # obj.generate_all_train_image()


