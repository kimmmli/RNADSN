import pyreadr
import torch.utils.data
import os
import numpy as np
import torch

## Below is the cross-technique version (second version FICC\miCLIP use miclip as train and ifcc as test)
seed = 2
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def get_list_value(list):
    list_value = []
    for value in list.values():
        list_value.append(value)
    return np.asarray(list_value, dtype=np.int32).squeeze(0)


def get_onehot(array):
    onehot = np.eye(4)[array - 1]
    return onehot


ficc_trna_pos_token = get_list_value(pyreadr.read_r('data/FICC_tRNA_pos_token.rds'))
ficc_mrna_pos_token = get_list_value(pyreadr.read_r('data/FICC_mRNA_pos_token.rds'))
ficc_other_pos_token = get_list_value(pyreadr.read_r('data/FICC_other_pos_token.rds'))
trna_neg_token = get_list_value(pyreadr.read_r('data/trna_neg_token.rds'))
mrna_neg_token = get_list_value(pyreadr.read_r('data/mrna_neg_token.rds'))
other_neg_token = get_list_value(pyreadr.read_r('data/other_neg_token.rds'))
miclip_trna_pos_token=get_list_value(pyreadr.read_r('data/miCLIP_tRNA_pos_token.rds'))
miclip_other_pos_token=get_list_value(pyreadr.read_r('data/miCLIP_other_pos_token.rds'))
miclip_mrna_pos_token=get_list_value(pyreadr.read_r('data/miCLIP_mRNA_pos_token.rds'))

# get mid 41 one-hot
ficc_mrna_pos = get_onehot(ficc_mrna_pos_token[:, 480:521]) #162
mrna_neg = get_onehot(mrna_neg_token[:, 480:521])  # 4570
ficc_trna_pos = get_onehot(ficc_trna_pos_token[:, 480:521]) #121
trna_neg = get_onehot(trna_neg_token[:, 480:521])  # 4339
ficc_other_pos = get_onehot(ficc_other_pos_token[:, 480:521]) # 667
other_neg = get_onehot(other_neg_token[:, 480:521])  # 21630
miclip_other_pos = get_onehot(miclip_other_pos_token[:, 480:521])#1265
miclip_mrna_pos = get_onehot(miclip_mrna_pos_token[:, 480:521])#262
miclip_trna_pos = get_onehot(miclip_trna_pos_token[:, 480:521])#698


#
# np.random.shuffle(miclip_mrna_pos)
# np.random.shuffle(ficc_mrna_pos)
# np.random.shuffle(mrna_neg)
#
# # mrna_pos_test = mrna_pos[0:77, :, :]
# # mrna_neg_test = mrna_neg[0:770, :, :]
# # mrna_pos_train = mrna_pos[77:457, :, :]
# # mrna_neg_train = mrna_neg[770:4570, :, :]
#
# mrna_pos_test = np.load('mrna_pos_test.npy')
# mrna_neg_test = np.load('mrna_neg_test.npy')
# mrna_pos_train = np.load('mrna_pos_train.npy')
# mrna_neg_train = np.load('mrna_neg_train.npy')

# target:mrna source:other_trna train:miclip test:ficc
target_test = np.vstack((ficc_mrna_pos, mrna_neg[0:162,:]))
target_train = np.vstack(
    (miclip_mrna_pos,miclip_mrna_pos,miclip_mrna_pos,
     miclip_mrna_pos,miclip_mrna_pos,miclip_mrna_pos,
     miclip_mrna_pos,miclip_mrna_pos,miclip_mrna_pos,
     miclip_mrna_pos, mrna_neg[163:2570]))

target_test_label = torch.cat(
    (torch.ones(ficc_mrna_pos.shape[0], 1),
     torch.zeros(mrna_neg[0:162,:].shape[0], 1))
    , dim=0
)

target_train_label = torch.cat(
    (torch.ones(miclip_mrna_pos.shape[0] * 10, 1),
     torch.zeros(mrna_neg[163:2570].shape[0], 1))
    , dim=0
)

source_pos = np.vstack((miclip_trna_pos, miclip_trna_pos, miclip_other_pos))
source_neg = np.vstack((trna_neg[0:1400,:], other_neg[0:1400,:]))

np.random.shuffle(source_pos)
np.random.shuffle(source_neg)
#
# source_pos = np.vstack((trna_pos, trna_pos, trna_pos, trna_pos,
#                         other_pos, other_pos, other_pos, other_pos, other_pos,
#                         other_pos, other_pos, other_pos, other_pos, other_pos))
# source_neg = np.vstack((trna_neg, other_neg))

source_train = np.vstack((source_pos, source_neg))

source_train_label = torch.cat(
    (torch.ones(source_pos.shape[0], 1), torch.zeros(source_neg.shape[0], 1))
    , dim=0
)

target_train = torch.tensor(np.expand_dims(target_train, axis=1), dtype=torch.float32).type(torch.FloatTensor)
source_train = torch.tensor(np.expand_dims(source_train, axis=1), dtype=torch.float32).type(torch.FloatTensor)
target_test = torch.tensor(np.expand_dims(target_test, axis=1), dtype=torch.float32).type(torch.FloatTensor)


dataset_target = torch.utils.data.TensorDataset(target_train, target_train_label)
dataset_target_test = torch.utils.data.TensorDataset(target_test, target_test_label)
dataset_source = torch.utils.data.TensorDataset(source_train, source_train_label)

print(target_train.shape)
print(target_train_label.shape)
print(target_test.shape)
print(target_test_label.shape)
print('---------------------')
print(source_train.shape)
print(source_train_label.shape)


#
# # Below is the cross-cell version (hel293 training & hap1 testing)
#
# seed = 2
# np.random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
#
#
# def get_list_value(list):
#     list_value = []
#     for value in list.values():
#         list_value.append(value)
#     return np.asarray(list_value, dtype=np.int32).squeeze(0)
#
#
# def get_onehot(array):
#     onehot = np.eye(4)[array - 1]
#     return onehot
#
#
# hap1_trna_pos_token = get_list_value(pyreadr.read_r('data/HAP1_tRNA_pos_token.rds'))
# hap1_mrna_pos_token = get_list_value(pyreadr.read_r('data/HAP1_mRNA_pos_token.rds'))
# hap1_other_pos_token = get_list_value(pyreadr.read_r('data/HAP1_other_pos_token.rds'))
# trna_neg_token = get_list_value(pyreadr.read_r('data/trna_neg_token.rds'))
# mrna_neg_token = get_list_value(pyreadr.read_r('data/mrna_neg_token.rds'))
# other_neg_token = get_list_value(pyreadr.read_r('data/other_neg_token.rds'))
# hek293_trna_pos_token=get_list_value(pyreadr.read_r('data/HEK293_tRNA_pos_token.rds'))
# hek293_other_pos_token=get_list_value(pyreadr.read_r('data/HEK293_other_pos_token.rds'))
# hek293_mrna_pos_token=get_list_value(pyreadr.read_r('data/HEK293_mRNA_pos_token.rds'))
#
# # get mid 41 one-hot
# hap1_mrna_pos = get_onehot(hap1_mrna_pos_token[:, 480:521]) #90
# mrna_neg = get_onehot(mrna_neg_token[:, 480:521])  # 4570
# hap1_trna_pos = get_onehot(hap1_trna_pos_token[:, 480:521]) #46
# trna_neg = get_onehot(trna_neg_token[:, 480:521])  # 4339
# hap1_other_pos = get_onehot(hap1_other_pos_token[:, 480:521]) # 361
# other_neg = get_onehot(other_neg_token[:, 480:521])  # 21630
# hek293_other_pos = get_onehot(hek293_other_pos_token[:, 480:521])#1427
# hek293_mrna_pos = get_onehot(hek293_mrna_pos_token[:, 480:521])#302
# hek293_trna_pos = get_onehot(hek293_trna_pos_token[:, 480:521])#738
#
#
#
# np.random.shuffle(hap1_mrna_pos)
# np.random.shuffle(hek293_mrna_pos)
# np.random.shuffle(mrna_neg)
# #
# # # mrna_pos_test = mrna_pos[0:77, :, :]
# # # mrna_neg_test = mrna_neg[0:770, :, :]
# # # mrna_pos_train = mrna_pos[77:457, :, :]
# # # mrna_neg_train = mrna_neg[770:4570, :, :]
# #
# # mrna_pos_test = np.load('mrna_pos_test.npy')
# # mrna_neg_test = np.load('mrna_neg_test.npy')
# # mrna_pos_train = np.load('mrna_pos_train.npy')
# # mrna_neg_train = np.load('mrna_neg_train.npy')
#
# # target:mrna source:other_trna train:hek293 test:hap1
# target_test = np.vstack((hap1_mrna_pos, mrna_neg[0:90,:]))
# target_train = np.vstack(
#     (hek293_mrna_pos,hek293_mrna_pos, hek293_mrna_pos,
#      hek293_mrna_pos,hek293_mrna_pos,hek293_mrna_pos,
#      hek293_mrna_pos,hek293_mrna_pos,hek293_mrna_pos,
#      hek293_mrna_pos,mrna_neg[90:3110]))
#
# target_test_label = torch.cat(
#     (torch.ones(hap1_mrna_pos.shape[0], 1),
#      torch.zeros(mrna_neg[0:90,:].shape[0], 1))
#     , dim=0
# )
#
# target_train_label = torch.cat(
#     (torch.ones(hek293_mrna_pos.shape[0] * 10, 1),
#      torch.zeros(mrna_neg[90:3110].shape[0], 1))
#     , dim=0
# )
#
# source_pos = np.vstack((hek293_trna_pos, hek293_trna_pos, hek293_other_pos))
# source_neg = np.vstack((trna_neg[0:1450,:], other_neg[0:1450,:]))
#
# np.random.shuffle(source_pos)
# np.random.shuffle(source_neg)
# #
# # source_pos = np.vstack((trna_pos, trna_pos, trna_pos, trna_pos,
# #                         other_pos, other_pos, other_pos, other_pos, other_pos,
# #                         other_pos, other_pos, other_pos, other_pos, other_pos))
# # source_neg = np.vstack((trna_neg, other_neg))
#
# source_train = np.vstack((source_pos, source_neg))
#
# source_train_label = torch.cat(
#     (torch.ones(source_pos.shape[0], 1), torch.zeros(source_neg.shape[0], 1))
#     , dim=0
# )
#
# target_train = torch.tensor(np.expand_dims(target_train, axis=1), dtype=torch.float32).type(torch.FloatTensor)
# source_train = torch.tensor(np.expand_dims(source_train, axis=1), dtype=torch.float32).type(torch.FloatTensor)
# target_test = torch.tensor(np.expand_dims(target_test, axis=1), dtype=torch.float32).type(torch.FloatTensor)
#
#
# dataset_target = torch.utils.data.TensorDataset(target_train, target_train_label)
# dataset_target_test = torch.utils.data.TensorDataset(target_test, target_test_label)
# dataset_source = torch.utils.data.TensorDataset(source_train, source_train_label)
#
# print(target_train.shape)
# print(target_train_label.shape)
# print(target_test.shape)
# print(target_test_label.shape)
# print('---------------------')
# print(source_train.shape)
# print(source_train_label.shape)
#





## Below is the normal version (first version using cross validation)
# seed = 2
# np.random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
#
#
# def get_list_value(list):
#     list_value = []
#     for value in list.values():
#         list_value.append(value)
#     return np.asarray(list_value, dtype=np.int32).squeeze(0)
#
#
# def get_onehot(array):
#     onehot = np.eye(4)[array - 1]
#     return onehot
#
#
# trna_pos_token = get_list_value(pyreadr.read_r('data/trna_pos_token.rds'))
# trna_neg_token = get_list_value(pyreadr.read_r('data/trna_neg_token.rds'))
# mrna_pos_token = get_list_value(pyreadr.read_r('data/mrna_pos_token.rds'))
# mrna_neg_token = get_list_value(pyreadr.read_r('data/mrna_neg_token.rds'))
# other_pos_token = get_list_value(pyreadr.read_r('data/other_pos_token.rds'))
# other_neg_token = get_list_value(pyreadr.read_r('data/other_neg_token.rds'))
#
# # get mid 41 one-hot
# mrna_pos = get_onehot(mrna_pos_token[:, 480:521])  # 457
# mrna_neg = get_onehot(mrna_neg_token[:, 480:521])  # 4570
# trna_pos = get_onehot(trna_pos_token[:, 480:521])  # 1076
# trna_neg = get_onehot(trna_neg_token[:, 480:521])  # 4339
# other_pos = get_onehot(other_pos_token[:, 480:521])  # 2163
# other_neg = get_onehot(other_neg_token[:, 480:521])  # 21630
#
# np.random.shuffle(mrna_pos)
# np.random.shuffle(mrna_neg)
#
# # mrna_pos_test = mrna_pos[0:77, :, :]
# # mrna_neg_test = mrna_neg[0:770, :, :]
# # mrna_pos_train = mrna_pos[77:457, :, :]
# # mrna_neg_train = mrna_neg[770:4570, :, :]
#
# mrna_pos_test = np.load('mrna_pos_test.npy')
# mrna_neg_test = np.load('mrna_neg_test.npy')
# mrna_pos_train = np.load('mrna_pos_train.npy')
# mrna_neg_train = np.load('mrna_neg_train.npy')
#
# target_test = np.vstack((mrna_pos_test, mrna_neg_test))
# target_train = np.vstack(
#     (mrna_pos_train, mrna_pos_train, mrna_pos_train, mrna_pos_train,
#      mrna_pos_train, mrna_pos_train, mrna_pos_train, mrna_pos_train,
#      mrna_pos_train, mrna_pos_train, mrna_neg_train))
#
# target_test_label = torch.cat(
#     (torch.ones(mrna_pos_test.shape[0], 1),
#      torch.zeros(mrna_neg_test.shape[0], 1))
#     , dim=0
# )
#
# target_train_label = torch.cat(
#     (torch.ones(mrna_pos_train.shape[0] * 10, 1),
#      torch.zeros(mrna_neg_train.shape[0], 1))
#     , dim=0
# )
#
# source_pos = np.vstack((trna_pos, trna_pos, other_pos))
# source_neg = np.vstack((trna_neg, other_neg))
#
# np.random.shuffle(source_pos)
# np.random.shuffle(source_neg)
#
# source_pos = np.vstack((trna_pos, trna_pos, trna_pos, trna_pos,
#                         other_pos, other_pos, other_pos, other_pos, other_pos,
#                         other_pos, other_pos, other_pos, other_pos, other_pos))
# source_neg = np.vstack((trna_neg, other_neg))
#
# source_train = np.vstack((source_pos, source_neg))
#
# source_train_label = torch.cat(
#     (torch.ones(source_pos.shape[0], 1), torch.zeros(source_neg.shape[0], 1))
#     , dim=0
# )
#
# target_train = torch.tensor(np.expand_dims(target_train, axis=1), dtype=torch.float32).type(torch.FloatTensor)
# source_train = torch.tensor(np.expand_dims(source_train, axis=1), dtype=torch.float32).type(torch.FloatTensor)
# target_test = torch.tensor(np.expand_dims(target_test, axis=1), dtype=torch.float32).type(torch.FloatTensor)
#
#
# dataset_target = torch.utils.data.TensorDataset(target_train, target_train_label)
# dataset_target_test = torch.utils.data.TensorDataset(target_test, target_test_label)
# dataset_source = torch.utils.data.TensorDataset(source_train, source_train_label)
#
# print(target_train.shape)
# print(target_train_label.shape)
# print(target_test.shape)
# print(target_test_label.shape)
# print('---------------------')
# print(source_train.shape)
# print(source_train_label.shape)

