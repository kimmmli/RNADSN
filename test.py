import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from model_compat_withCNN import DSN
# from sequence_loader_forCNN import dataset_target_test
from torchmetrics.functional import average_precision, accuracy, auroc, f1_score, specificity


def test(epoch, dataset, fold):

    ###################
    # params          #
    ###################
    cuda = 1
    cudnn.benchmark = False
    ###################
    # load data       #
    ###################

    dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=len(dataset),
            shuffle=True
        )



    ####################
    # load model       #
    ####################

    my_net = DSN()
    checkpoint = torch.load(os.path.join('models/fold' + str(fold) + 'net' + str(epoch) + '.pth'))
    my_net.load_state_dict(checkpoint)
    my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    ####################
    # transform image  #
    ####################

    data_iter = iter(dataloader)

    data_input = data_iter.next()
    img, label = data_input

    batch_size = len(label)

    input_img = torch.FloatTensor(batch_size, 1, 41, 4)
    class_label = torch.LongTensor(batch_size)

    if cuda:
        img = img.cuda()
        label = label.cuda()
        input_img = input_img.cuda()
        class_label = class_label.cuda()

    input_img.resize_as_(input_img).copy_(img)
    class_label.resize_as_(label).copy_(label)
    inputv_img = Variable(input_img)
    classv_label = Variable(class_label)


    result = my_net(input_data=inputv_img, mode='source', rec_scheme='share')
    # print(result[3])

    accu = accuracy(result[3], label.int())
    auc = auroc(result[3], label.int())
    precision = average_precision(result[3], label, pos_label=1)
    f1 = f1_score(result[3], label.int())
    spec = specificity(result[3], label.int())
    # mcc = matthews_corrcoef(result[3], label.int(), num_classes=2)

    print('Test: epoch: %d, accuracy: %f, AUC: %f, pre: %f, f1: %f, spec: %f'
          % (epoch, accu, auc, precision, f1, spec))
    return precision




