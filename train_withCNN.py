import torch.utils.data
import os
import shutil
import pyreadr
from sklearn.model_selection import KFold
from model_compat_withCNN import DSN
from functions import SIMSE, DiffLoss, MSE
from test import test
# from sequence_loader_forCNN import dataset_target, dataset_source
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from torch.autograd import Variable

seed = 2
print(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


######################
# params             #
######################

model_root = 'model'
cuda = 1
cudnn.benchmark = True
lr = 0.005
batch_size = 64
n_epoch = 20
step_decay_weight = 0.9
lr_decay_step = 600
active_domain_loss_step = 100000
weight_decay = 1e-5


alpha_weight = 0.02
beta_weight = 0.075
gamma_weight = 0.25
momentum = 0.9

#######################
# load data         #
#######################

# <editor-fold desc="read from .rds">
def get_list_value(list):
    list_value = []
    for value in list.values():
        list_value.append(value)
    return np.asarray(list_value, dtype=np.int32).squeeze(0)


def get_onehot(array):
    onehot = np.eye(4)[array - 1]
    return onehot

trna_pos_token = get_list_value(pyreadr.read_r('data/trna_pos_token.rds'))
trna_neg_token = get_list_value(pyreadr.read_r('data/trna_neg_token.rds'))
mrna_pos_token = get_list_value(pyreadr.read_r('data/mrna_pos_token.rds'))
mrna_neg_token = get_list_value(pyreadr.read_r('data/mrna_neg_token.rds'))
other_pos_token = get_list_value(pyreadr.read_r('data/other_pos_token.rds'))
other_neg_token = get_list_value(pyreadr.read_r('data/other_neg_token.rds'))
# </editor-fold>

# get mid 41 one-hot
mrna_pos = get_onehot(mrna_pos_token[:, 480:521])  # 457
mrna_neg = get_onehot(mrna_neg_token[:, 480:521])  # 4570
trna_pos = get_onehot(trna_pos_token[:, 480:521])  # 1076
trna_neg = get_onehot(trna_neg_token[:, 480:521])  # 4339
other_pos = get_onehot(other_pos_token[:, 480:521])  # 2163
other_neg = get_onehot(other_neg_token[:, 480:521])  # 21630

# <editor-fold desc="Load source data">
source_pos = np.vstack((trna_pos, trna_pos, other_pos))
source_neg = np.vstack((trna_neg, other_neg))

np.random.shuffle(source_pos)
np.random.shuffle(source_neg)

source_pos = np.vstack((trna_pos, trna_pos, trna_pos, trna_pos,
                        other_pos, other_pos, other_pos, other_pos, other_pos,
                        other_pos, other_pos, other_pos, other_pos, other_pos))
source_neg = np.vstack((trna_neg, other_neg))

source_train = np.vstack((source_pos, source_neg))

source_train_label = torch.cat(
    (torch.ones(source_pos.shape[0], 1), torch.zeros(source_neg.shape[0], 1))
    , dim=0
)

source_train = torch.tensor(np.expand_dims(source_train, axis=1), dtype=torch.float32).type(torch.FloatTensor)

dataset_source = torch.utils.data.TensorDataset(source_train, source_train_label)
# </editor-fold>

dataloader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=batch_size,
    shuffle=True
)

kf = KFold(n_splits = 6, shuffle=True, random_state=0)
# for train_index, test_index in kf.split(mrna_pos):     # 将数据划分为k折
#     mrna_pos_train = mrna_pos[train_index]   # 选取的训练集数据下标
#     mrna_pos_test  = mrna_pos[test_index]          # 选取的测试集数据下标
#     tr = str('mrna_pos_train_')+ str(z) + str('.npy')
#     te = str('mrna_pos_test_') + str(z) + str('.npy')
#     np.save(tr, mrna_pos_train)
#     np.save(te, mrna_pos_test)
#     z+=1
mrna_pos_train = np.load('mrna_pos_train_6.npy')
mrna_pos_test = np.load('mrna_pos_test_6.npy')

z = 31
for train_index, test_index in kf.split(mrna_neg):
    print("this is the fold %d" % (z))
    if not os.path.exists('/home/jinge/DSN/models'):
        os.mkdir('/home/jinge/DSN/models')
    #####################mrna_pos_train = np.load('mrna_pos_train_1.npy')
    #  load model       #
    #####################

    my_net = DSN()


    #####################
    # setup optimizer   #
    #####################

    def exp_lr_scheduler(optimizer, step, init_lr=lr, lr_decay_step=lr_decay_step,
                         step_decay_weight=step_decay_weight):
        # Decay learning rate by a factor of step_decay_weight every lr_decay_step
        current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))

        if step % lr_decay_step == 0:
            print('learning rate is set to %f' % current_lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        return optimizer


    optimizer = optim.SGD(my_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    loss_classification = torch.nn.BCELoss()
    loss_recon1 = MSE()
    loss_recon2 = SIMSE()
    loss_diff = DiffLoss()
    loss_similarity = torch.nn.CrossEntropyLoss()

    if cuda:
        my_net = my_net.cuda()
        loss_classification = loss_classification.cuda()
        loss_recon1 = loss_recon1.cuda()
        loss_recon2 = loss_recon2.cuda()
        loss_diff = loss_diff.cuda()
        loss_similarity = loss_similarity.cuda()

    for p in my_net.parameters():
        p.requires_grad = True

    #######################
    # results           #
    #######################
    max_target_pre = 0
    cor_epoch = 0

    mrna_neg_train = mrna_neg[train_index]  # 选取的训练集数据下标
    mrna_neg_test = mrna_neg[test_index]  # 选取的测试集数据下标
    target_test = np.vstack((mrna_pos_test, mrna_neg_test))
    target_train = np.vstack(
        (mrna_pos_train, mrna_pos_train, mrna_pos_train, mrna_pos_train,
         mrna_pos_train, mrna_pos_train, mrna_pos_train, mrna_pos_train,
         mrna_pos_train, mrna_pos_train, mrna_neg_train))

    target_test_label = torch.cat(
        (torch.ones(mrna_pos_test.shape[0], 1),
         torch.zeros(mrna_neg_test.shape[0], 1))
        , dim=0
    )

    target_train_label = torch.cat(
        (torch.ones(mrna_pos_train.shape[0] * 10, 1),
         torch.zeros(mrna_neg_train.shape[0], 1))
        , dim=0
    )

    target_train = torch.tensor(np.expand_dims(target_train, axis=1), dtype=torch.float32).type(torch.FloatTensor)
    target_test = torch.tensor(np.expand_dims(target_test, axis=1), dtype=torch.float32).type(torch.FloatTensor)

    dataset_target = torch.utils.data.TensorDataset(target_train, target_train_label)
    dataset_target_test = torch.utils.data.TensorDataset(target_test, target_test_label)


    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True
    )

    #############################
    # training network          #
    #############################

    broadcast_buffers = False
    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    dann_epoch = np.floor(active_domain_loss_step / len_dataloader * 1.0)

    current_step = 0
    for epoch in range(n_epoch):

        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        i = 0

        while i < len_dataloader:

            ###################################
            # target data training            #
            ###################################

            data_target = data_target_iter.next()
            t_img, t_label = data_target

            my_net.zero_grad()
            loss = 0
            batch_size = len(t_label)

            input_img = torch.FloatTensor(batch_size, 1, 41, 4)
            class_label = torch.LongTensor(batch_size)
            domain_label = torch.ones(batch_size)
            domain_label = domain_label.long()

            t_img = t_img.float()
            t_label = t_label.float()
            input_img = input_img.float()
            class_label = class_label.float()
            domain_label = domain_label.float()

            if cuda:
                t_img = t_img.cuda()
                t_label = t_label.cuda()
                input_img = input_img.cuda()
                class_label = class_label.cuda()
                domain_label = domain_label.cuda()

            input_img.resize_as_(t_img).copy_(t_img)
            class_label.resize_as_(t_label).copy_(t_label)
            target_inputv_img = Variable(input_img)
            target_classv_label = Variable(class_label)
            target_domainv_label = Variable(domain_label)

            if current_step > active_domain_loss_step:
                p = float(i + (epoch - dann_epoch) * len_dataloader / (n_epoch - dann_epoch) / len_dataloader)
                p = 2. / (1. + np.exp(-10 * p)) - 1

                # activate domain loss
                result = my_net(input_data=target_inputv_img, mode='target', rec_scheme='all', p=p)
                target_privte_code, target_share_code, target_domain_label, target_class_label, target_rec_code = result
                target_dann = gamma_weight * loss_similarity(target_domain_label, target_domainv_label)
                loss = loss + target_dann
            else:
                target_dann = Variable(torch.zeros(1).float())
                result = my_net(input_data=target_inputv_img, mode='target', rec_scheme='all')
                target_privte_code, target_share_code, _, target_class_label, target_rec_code = result

            target_classification = loss_classification(target_class_label.to(torch.float), target_classv_label.to(torch.float))
            loss = loss + target_classification * 8

            target_diff = beta_weight * loss_diff(target_privte_code, target_share_code)
            loss = loss + target_diff
            target_mse = alpha_weight * loss_recon1(target_rec_code, target_inputv_img)
            loss = loss + target_mse
            target_simse = alpha_weight * loss_recon2(target_rec_code, target_inputv_img)
            loss = loss + target_simse

            loss.backward()

            optimizer.step()

            ###################################
            # source data training            #
            ###################################

            data_source = data_source_iter.next()
            s_img, s_label = data_source

            my_net.zero_grad()
            batch_size = len(s_label)

            input_img = torch.FloatTensor(batch_size, 1, 41, 4)
            class_label = torch.LongTensor(batch_size)
            domain_label = torch.zeros(batch_size)
            domain_label = domain_label.long()

            loss = 0

            if cuda:
                s_img = s_img.cuda()
                s_label = s_label.cuda()
                input_img = input_img.cuda()
                class_label = class_label.cuda()
                domain_label = domain_label.cuda()

            s_label = s_label.long()

            input_img.resize_as_(input_img).copy_(s_img)
            class_label.resize_as_(s_label).copy_(s_label)
            source_inputv_img = Variable(input_img)
            source_classv_label = Variable(class_label)
            source_domainv_label = Variable(domain_label)

            if current_step > active_domain_loss_step:

                # activate domain loss

                result = my_net(input_data=source_inputv_img, mode='source', rec_scheme='all', p=p)
                source_privte_code, source_share_code, source_domain_label, source_class_label, source_rec_code = result
                source_dann = gamma_weight * loss_similarity(source_domain_label, source_domainv_label)
                loss = loss + source_dann
            else:
                source_dann = Variable(torch.zeros(1).float())
                result = my_net(input_data=source_inputv_img, mode='source', rec_scheme='all')
                source_privte_code, source_share_code, _, source_class_label, source_rec_code = result

            source_classification = loss_classification(source_class_label.to(torch.float), source_classv_label.to(torch.float))

            loss = loss + source_classification

            source_diff = beta_weight * loss_diff(source_privte_code, source_share_code)
            loss = loss + source_diff
            source_mse = alpha_weight * loss_recon1(source_rec_code, source_inputv_img)
            loss = loss + source_mse
            source_simse = alpha_weight * loss_recon2(source_rec_code, source_inputv_img)
            loss = loss + source_simse

            loss.backward()
            optimizer = exp_lr_scheduler(optimizer=optimizer, step=current_step)
            optimizer.step()

            i = i + 1
            current_step = current_step + 1

        print('Train specific loss: target_classification: %f,source_classification: %f, \n source_dann: %f, source_diff: %f, ' \
              'source_mse: %f, source_simse: %f, \n target_dann: %f, target_diff: %f, ' \
              'target_mse: %f, target_simse: %f' \
              % (target_classification.data.cpu().numpy(), source_classification.data.cpu().numpy(), source_dann.data.cpu().numpy(),
                 source_diff.data.cpu().numpy(),
                 source_mse.data.cpu().numpy(), source_simse.data.cpu().numpy(), target_dann.data.cpu().numpy(),
                 target_diff.data.cpu().numpy(), target_mse.data.cpu().numpy(), target_simse.data.cpu().numpy()))
        print('step: %d, loss: %f' % (current_step, loss.cpu().data.numpy()))


        torch.save(my_net.state_dict(), 'models/fold' + str(z) + 'net' + str(epoch) + '.pth')
        precision = test(epoch=epoch, dataset=dataset_target_test, fold = z)

        if precision > max_target_pre:
            max_target_pre = precision
            cor_epoch = epoch
        print('Current maximum pre: %f, epoch: %d' % (max_target_pre, cor_epoch) )

        print("-------------------------------------------------------------------------------------")
    print('done')
    z +=1




