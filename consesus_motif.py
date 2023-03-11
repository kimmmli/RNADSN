import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from model_compat_withCNN import DSN





my_net = DSN()
checkpoint = torch.load(os.path.join('models/fold' + str(fold) + 'net' + str(epoch) + '.pth'))
my_net.load_state_dict(checkpoint)