import torch.nn as nn
from functions import ReverseLayerF
import torch

class DSN(nn.Module):
    def __init__(self, code_size=50, n_class=1):
        super(DSN, self).__init__()
        self.code_size = code_size

        ##########################################
        # private source encoder
        ##########################################
        self.source_encoder_conv = nn.Sequential()
        self.source_encoder_conv.add_module('pse_conv1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(2,1), padding=2))
        self.source_encoder_conv.add_module('pse_BN1', nn.BatchNorm2d(6))
        self.source_encoder_conv.add_module('pse_flatten1', nn.Flatten(start_dim=1, end_dim=2))
        self.source_encoder_gru = nn.Sequential()
        self.source_encoder_gru.add_module('pse_gru1', nn.LSTM(8, 8, batch_first=True, bidirectional=True))
        self.source_encoder_bn = nn.Sequential()
        self.source_encoder_bn.add_module('pse_BN', nn.BatchNorm1d(264))
        self.source_encoder_bn.add_module('pse_flatten2', nn.Flatten(start_dim=1))


        self.source_encoder_fc = nn.Sequential()
        self.source_encoder_fc.add_module('pse_Linear1', nn.Linear(4224, 1024))
        self.source_encoder_fc.add_module('pse_relu1', nn.ReLU(True))
        self.source_encoder_fc.add_module('pse_dropout1', nn.Dropout(p=0.2))
        self.source_encoder_fc.add_module('pse_Linear2', nn.Linear(1024, 256))
        self.source_encoder_fc.add_module('pse_relu2', nn.ReLU(True))
        self.source_encoder_fc.add_module('pse_dropout2', nn.Dropout(p=0.2))
        self.source_encoder_fc.add_module('pse_Linear3', nn.Linear(256, code_size))

        #########################################
        # private target encoder
        #########################################

        self.target_encoder_conv = nn.Sequential()
        self.target_encoder_conv.add_module('pte_conv1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(2,1), padding=2))
        self.target_encoder_conv.add_module('pte_BN1', nn.BatchNorm2d(6))
        self.target_encoder_conv.add_module('pte_flatten1', nn.Flatten(start_dim=1, end_dim=2))
        self.target_encoder_gru = nn.Sequential()
        self.target_encoder_gru.add_module('pte_gru1', nn.LSTM(8, 8, batch_first=True, bidirectional=True))
        self.target_encoder_bn = nn.Sequential()
        self.target_encoder_bn.add_module('pte_BN', nn.BatchNorm1d(264))
        self.target_encoder_bn.add_module('pte_flatten2', nn.Flatten(start_dim=1))

        self.target_encoder_fc = nn.Sequential()
        self.target_encoder_fc.add_module('pte_Linear1', nn.Linear(4224, 1024))
        self.target_encoder_fc.add_module('pte_relu1', nn.ReLU(True))
        self.target_encoder_fc.add_module('pte_dropout1', nn.Dropout(p=0.2))
        self.target_encoder_fc.add_module('pte_Linear2', nn.Linear(1024, 256))
        self.target_encoder_fc.add_module('pte_relu2', nn.ReLU(True))
        self.target_encoder_fc.add_module('pte_dropout2', nn.Dropout(p=0.2))
        self.target_encoder_fc.add_module('pte_Linear3', nn.Linear(256, code_size))

        ################################
        # shared encoder (dann_mnist)
        ################################

        self.shared_encoder_conv = nn.Sequential()
        self.shared_encoder_conv.add_module('se_conv1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(2,1), padding=2))
        self.shared_encoder_conv.add_module('se_BN1', nn.BatchNorm2d(6))
        self.shared_encoder_conv.add_module('se_flatten1', nn.Flatten(start_dim=1, end_dim=2))
        self.shared_encoder_gru = nn.Sequential()
        self.shared_encoder_gru.add_module('se_gru1', nn.LSTM(8, 8, batch_first=True, bidirectional=True))
        self.shared_encoder_bn = nn.Sequential()
        self.shared_encoder_bn.add_module('se_BN', nn.BatchNorm1d(264))
        self.shared_encoder_bn.add_module('se_flatten2', nn.Flatten(start_dim=1))

        self.shared_encoder_fc = nn.Sequential()
        self.shared_encoder_fc.add_module('se_Linear1', nn.Linear(4224, 1024))
        self.shared_encoder_fc.add_module('se_relu1', nn.ReLU(True))
        self.shared_encoder_fc.add_module('se_dropout1', nn.Dropout(p=0.2))
        self.shared_encoder_fc.add_module('se_Linear2', nn.Linear(1024, 256))
        self.shared_encoder_fc.add_module('se_relu2', nn.ReLU(True))
        self.shared_encoder_fc.add_module('se_dropout2', nn.Dropout(p=0.2))
        self.shared_encoder_fc.add_module('se_Linear3', nn.Linear(256, code_size))

        # classify
        self.shared_encoder_pred_class = nn.Sequential()
        self.shared_encoder_pred_class.add_module('fc_se4', nn.Linear(in_features=code_size, out_features=16))
        self.shared_encoder_pred_class.add_module('relu_se4', nn.ReLU(True))
        self.shared_encoder_pred_class.add_module('fc_se5', nn.Linear(in_features=16, out_features=n_class))

        self.shared_encoder_pred_domain = nn.Sequential()
        self.shared_encoder_pred_domain.add_module('fc_se6', nn.Linear(in_features=50, out_features=16))
        self.shared_encoder_pred_domain.add_module('relu_se6', nn.ReLU(True))

        # classify two domain
        self.shared_encoder_pred_domain.add_module('fc_se7', nn.Linear(in_features=16, out_features=1))

        ######################################
        # shared decoder (small decoder)
        ######################################

        self.shared_decoder_fc = nn.Sequential()
        self.shared_decoder_fc.add_module('fc_sd1', nn.Linear(in_features=code_size, out_features=164))
        # self.shared_decoder_fc.add_module('relu_sd1', nn.ReLU(True))

        self.shared_decoder_conv = nn.Sequential()
        self.shared_decoder_conv.add_module('relu_sd2', nn.ReLU(True))



    def forward(self, input_data, mode, rec_scheme, p=0.0):

        result = []

        if mode == 'source':

            # source private encoder
            private_feat = self.source_encoder_conv(input_data)
            private_feat, _ = self.source_encoder_gru(private_feat)
            private_feat = self.source_encoder_bn(private_feat)
            private_feat = private_feat.view(-1, 4224)
            private_code = self.source_encoder_fc(private_feat)

        elif mode == 'target':

            private_feat = self.target_encoder_conv(input_data)
            private_feat, _ = self.target_encoder_gru(private_feat)
            private_feat = self.target_encoder_bn(private_feat)
            private_feat = private_feat.view(-1, 4224)
            private_code = self.source_encoder_fc(private_feat)

        result.append(private_code)

        # shared encoder
        shared_feat = self.shared_encoder_conv(input_data)
        shared_feat, _ = self.shared_encoder_gru(shared_feat)
        shared_feat = self.shared_encoder_bn(shared_feat)
        shared_feat = shared_feat.view(-1, 4224)
        shared_code = self.shared_encoder_fc(shared_feat)
        result.append(shared_code)


        reversed_shared_code = ReverseLayerF.apply(shared_code, p)
        domain_label = self.shared_encoder_pred_domain(reversed_shared_code)
        result.append(domain_label)


        class_label = torch.sigmoid(self.shared_encoder_pred_class(shared_code))
        result.append(class_label)

        # shared decoder

        if rec_scheme == 'share':
            union_code = shared_code
        elif rec_scheme == 'all':
            union_code = private_code + shared_code
        elif rec_scheme == 'private':
            union_code = private_code

        rec_vec = self.shared_decoder_fc(union_code)
        rec_vec = rec_vec.view(-1, 41, 4)

        rec_code = self.shared_decoder_conv(rec_vec)
        result.append(rec_code)

        return result





