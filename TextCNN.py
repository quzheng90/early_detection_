from utils import to_var
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Function
import config
import argparse
import torch.nn.functional as F

parse = argparse.ArgumentParser()
parser = config.parse_arguments(parse)
# train = '../Data/weibo/train.pickle'
# test = '../Data/weibo/test.pickle'
# output = '../Data/weibo/output/'
#args = parser.parse_args([train, test, output])
args=parser.parse_args()

class ReverseLayerF(Function):
    #def __init__(self, lambd):
        #self.lambd = lambd

    @staticmethod
    def forward(self, x):
        self.lambd = args.lambd
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x):
    return ReverseLayerF().apply(x)

# Neural Network Model (1 hidden layer)
class CNN_Fusion(nn.Module):
    def __init__(self, args, W):
        super(CNN_Fusion, self).__init__()
        self.args = args

        self.event_num = args.event_num

        vocab_size = args.vocab_size
        emb_dim = args.embed_dim

        C = args.class_num
        self.hidden_size = args.hidden_dim
        self.lstm_size = args.embed_dim
        self.social_size = 19

        # TEXT RNN

        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.embed.weight = nn.Parameter(torch.from_numpy(W))
        self.lstm = nn.LSTM(self.lstm_size, self.lstm_size)
        self.text_fc = nn.Linear(self.lstm_size, self.hidden_size)
        self.text_encoder = nn.Linear(emb_dim, self.hidden_size)

        ### TEXT CNN
        channel_in = 1
        filter_num = 20
        window_size = [1, 2, 3, 4]
        self.convs = nn.ModuleList([nn.Conv2d(channel_in, filter_num, (K, emb_dim)) for K in window_size])
        self.fc1 = nn.Linear(len(window_size) * filter_num, self.hidden_size)

        self.dropout = nn.Dropout(args.dropout)

        ###social context
        self.social = nn.Linear(self.social_size, self.hidden_size)

        ##ATTENTION
        self.attention_layer = nn.Linear(self.hidden_size, emb_dim)

        ##User Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1',  nn.Linear(self.hidden_size, 256))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(256))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout1d())
        self.class_classifier.add_module('tc_l1', nn.Linear(256,64))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(64))
        self.class_classifier.add_module('c_re', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout1d())
        self.class_classifier.add_module('tc_l2', nn.Linear(64,32))
        self.class_classifier.add_module('c_bn3', nn.BatchNorm1d(32))
        self.class_classifier.add_module('tc_re1', nn.ReLU(True))
        self.class_classifier.add_module('tc_nn', nn.Dropout1d())
        self.class_classifier.add_module('tc_l3', nn.Linear(32,self.hidden_size))
        self.class_classifier.add_module('tc_bn1', nn.BatchNorm1d(self.hidden_size))
        self.class_classifier.add_module('tc_re2', nn.ReLU(True))
        self.class_classifier.add_module('tc_fc1', nn.Linear(self.hidden_size, 2))
        self.class_classifier.add_module('tc_sm', nn.Softmax(dim=1))

        ###Domain Classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.hidden_size, self.hidden_size))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(self.hidden_size))
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.hidden_size, self.event_num))
        self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))


    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (to_var(torch.zeros(1, batch_size, self.lstm_size)),
                to_var(torch.zeros(1, batch_size, self.lstm_size)))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (sample number,hidden_dim, length)
        #x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def forward(self, text, mask):
        
        #########CNN##################
        text = self.embed(text)
        text = text * mask.unsqueeze(2).expand_as(text)
        text = text.unsqueeze(1)
        text = [F.relu(conv(text)).squeeze(3) for conv in self.convs]  # [(N,hidden_dim,W), ...]*len(window_size)
        #text = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in text]  # [(N,hidden_dim), ...]*len(window_size)
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]
        text = torch.cat(text, 1)
        text = F.relu(self.fc1(text))
        #text = self.dropout(text)

        ### Class
        #class_output = self.class_classifier(text_image)
        class_output = self.class_classifier(text)
        ## Domain
        reverse_feature = grad_reverse(text)
        domain_output = self.domain_classifier(reverse_feature)
     
        return class_output, domain_output