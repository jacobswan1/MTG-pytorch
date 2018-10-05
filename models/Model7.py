"""Model7 is for semantic embedding & attention. We replace the global classification with the semantic classification,
thus applicable for textual grounding problem."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import sys
sys.path.insert(0, '/../')
import torch
import numpy as np
import torch.nn as nn
from lib.configure.config import Config
from lib.resnet.resnet import resnet50
from torch.autograd import Variable
from lib.bilinear_pooling.CompactBilinearPooling import CompactBilinearPooling


class Model7(nn.Module):

    def __init__(self, opts, body_pretrain=False):
        super(Model7, self).__init__()

        # Load pre-trained back-boned model
        print('==> Building backbone model...')
        config = Config()
        config.IMAGES_PER_GPU = opts.batch_size
        config.NUM_CLASSES = opts.class_num

        # Load Attribute module
        attr_branch = AttributeBranch(300)
        attr_res_net = resnet50(True, path='./checkpoint/AENet_clsfier_person_256d_4.pth', classnum=4)

        # Load semantic embeddings
        dictionary = {'man': [1, 0, 0.5, 0.5],
                      'woman': [0, 1, 0.5, 0.5],
                      'lady': [0, 1, 0.25, 0.75],
                      'female': [0, 1, 0.5, 0.5],
                      'boy': [1, 0, 1, 0],
                      'girl': [0, 1, 1, 0],
                      'kid': [0.5, 0.5, 1, 0],
                      'child': [0.5, 0.5, 1, 0],
                      'young': [0.5, 0.5, 1, 0],
                      'elderly': [0.5, 0.5, 0, 1]}
        for key in dictionary.keys():
            dictionary[key] = np.asarray(dictionary[key])

        # Freeze the attr-resnet model
        for param in attr_res_net.parameters():
            param.requires_grad = False

        for param in attr_res_net.fc.parameters():
            param.requires_grad = False

        # Freeze the attribute branch or not
        for param in attr_branch.parameters():
            param.requires_grad = True

        self.attr_branch = attr_branch
        self.opts = opts
        self.attr_res_net = attr_res_net
        self.pool = nn.AvgPool2d(kernel_size=64, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.regressor = nn.Linear(256, 4)
        self.semantic_layer = SemanticLayer(dictionary)

    def forward(self, img, label, embeddings):

        # Attribute Branch
        conv_feat4, conv_feat = self.attr_res_net(img)
        attr_map, att_conv_feature = self.attr_branch(conv_feat, embeddings)
        feat = self.pool(att_conv_feature)
        feat = self.regressor(feat.view(feat.shape[0], feat.shape[1]))
        output = self.semantic_layer(feat, label)
        return output, attr_map, att_conv_feature


class AttributeBranch(nn.Module):

    def __init__(self, attr_num):
        super(AttributeBranch, self).__init__()

        self.textual_emb = nn.Linear(attr_num, 256)
        self.conv = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.mcb_attr = CompactBilinearPooling(256, 256, 256).cuda()
        self.mcb_conv1_attr = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0, bias=True)
        self.mcb_relu1_attr = nn.ReLU(inplace=True)
        self.mcb_conv2_attr = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.mcb_sigmoid = nn.Sigmoid()

    def forward(self, entity_feature, attr_one_hot):

        feature = self.mcb_relu1_attr(entity_feature)
        # Reshape attribute one hot input
        attr_one_hot = self.textual_emb(attr_one_hot)
        attr_one_hot = attr_one_hot.view(attr_one_hot.shape[0], attr_one_hot.shape[1], 1, 1)

        # stack attention map generating for P3, P4, P5
        attr_one_hot = attr_one_hot.expand_as(feature)

        # Attribute attention generation and applied
        mcb_attr_feat = self.mcb_attr(self.conv(attr_one_hot), feature)
        attr_map = self.mcb_sigmoid(self.mcb_conv2_attr(self.mcb_relu1_attr(self.mcb_conv1_attr(mcb_attr_feat))))
        attr_feature = (torch.mul(attr_map, entity_feature))

        return attr_map, attr_feature


class SemanticLayer(nn.Module):
    def __init__(self, dictionary):
        super(SemanticLayer, self).__init__()

        list_file = open('./others/low-level-attr.txt', 'r')
        entity_att = []
        for i in list_file.readlines():
            entity_att.append(i.replace('\n', ''))

        # Create semantic matrix
        s_matrix = torch.zeros(10, 4).cuda()
        for index, item in enumerate(entity_att):
            emb = torch.from_numpy(dictionary[item])
            s_matrix[index] = emb
        self.s_matrix = Variable(s_matrix)

    def forward(self, x, label):
        # x: (batch * 4)
        # label: (batch,)
        prob = Variable(torch.zeros(x.shape[0]))
        for index in range(x.shape[0]):
            lbl = label[index]
            prob[index] = torch.nn.functional.cosine_similarity(self.s_matrix, x[index].view(1, -1))[lbl]
        prob = prob.sum() / prob.shape[0]
        return 1-prob
