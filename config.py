import torch

# Is the PC has cuda
cuda_use = torch.cuda.is_available()
# which cuda to use
cuda_num = 2

# learning rate for D, the lr in Apple blog is 0.0001
d_lr = 0.001
# learning rate for R, the lr in Apple blog is 0.0001
r_lr = 0.001
# lambda in paper, the author of the paper said it's 0.01
delta = 0.0001
img_width = 55
img_height = 35
img_channels = 1

# synthetic image path
syn_path = 'dataset/SynthEyes_train_data'
# real image path
real_path = 'dataset/MPIIGaze_data'
# training result path to save
train_res_path = 'train_res'
# final_res_path = 'final_res'

# result show in 4 sample per line
pics_line = 4

# =================== training params ======================
# pre-train R times
r_pretrain = 1000
# pre-train D times
d_pretrain = 200
# train steps
train_steps = 10000000

batch_size = 128
# test_batch_size = 128
# the history buffer size
buffer_size = 12800
k_d = 1  # number of discriminator updates per step
k_r = 50  # number of generative network updates per step, the author of the paper said it's 50

# output R pre-training result per times
r_pre_per = 50
# output D pre-training result per times
d_pre_per = 50
# save model dictionary and training dataset output result per train times
save_per = 10


# pre-training dictionary path
# ref_pre_path = 'models/R_pre.pkl'
ref_pre_path = None
# disc_pre_path = 'models/D_pre.pkl'
disc_pre_path = None

# dictionary saving path
D_path = 'models/D_%d.pkl'
R_path = 'models/R_%d.pkl'

