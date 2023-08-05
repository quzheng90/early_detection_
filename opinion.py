import numpy as np
import argparse
import config
import time, os
# import random
#import process_data_weibo as process_data
from potential_user import Potantial_User
from utils import to_var, to_np, select, make_weights_for_balanced_classes, split_train_validation
import pickle
#import torchvision
#from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from word2vec_weibo import load_data
from sklearn import metrics
import scipy.io as sio
from TextCNN import CNN_Fusion
   


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser = config.parse_arguments(parse)
    args=parser.parse_args()
    print(args)
    print('loading data')
    train, validation, test, W = load_data(args)

    #train, validation = split_train_validation(train,  1)

    #weights = make_weights_for_balanced_classes(train[-1], 15)
    #weights = torch.DoubleTensor(weights)
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    train_dataset = Potantial_User(train)
    validate_dataset = Potantial_User(validation)
    test_dataset = Potantial_User(test) # not used
    # Data Loader (Input Pipeline)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)

    validate_loader = DataLoader(dataset = validate_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    print('building model')
    model = CNN_Fusion(args, W)

    if torch.cuda.is_available():
        print("CUDA")
        model.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()

    #Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, list(model.parameters())),
                                 lr= args.learning_rate)
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
                                 #lr= args.learning_rate)
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, list(model.parameters())),
                                  #lr=args.learning_rate)
    #scheduler = StepLR(optimizer, step_size= 10, gamma= 1)


    iter_per_epoch = len(train_loader)
    print("loader size " + str(len(train_loader)))
    best_validate_acc = 0.000
    best_loss = 100
    best_validate_dir = ''

    print('training model')
    adversarial = True
    # Train the Model
    for epoch in range(args.num_epochs):

        p = float(epoch) / 100
        #lambd = 2. / (1. + np.exp(-10. * p)) - 1
        lr = 1e-6

        optimizer.lr = lr
        #rgs.lambd = lambd

        start_time = time.time()
        cost_vector = []
        class_cost_vector = []
        domain_cost_vector = []
        acc_vector = []
        valid_acc_vector = []
        test_acc_vector = []
        vali_cost_vector = []
        test_cost_vector = []

        for i, (train_data, train_labels, event_labels) in enumerate(train_loader):
            train_text,  train_mask, train_labels, event_labels = \
                to_var(train_data[0]),  to_var(train_data[1]), \
                to_var(train_labels), to_var(event_labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()

            class_outputs, domain_outputs = model(train_text, train_mask)
            # ones = torch.ones(text_output.size(0))
            # ones_label = to_var(ones.type(torch.LongTensor))
            # zeros = torch.zeros(image_output.size(0))
            # zeros_label = to_var(zeros.type(torch.LongTensor))

            #modal_loss = criterion(text_output, ones_label)+ criterion(image_output, zeros_label)
            train_labels=train_labels.to(torch.long)
            class_loss = criterion(class_outputs, train_labels)
            event_labels=event_labels.to(torch.long)
            domain_loss = criterion(domain_outputs, event_labels)
            #减去一个比较小的loss

            loss = class_loss - 0.01*domain_loss
            loss.backward()
            optimizer.step()
            _, argmax = torch.max(class_outputs, 1)

            cross_entropy = True

            if True:
                accuracy = (train_labels == argmax.squeeze()).float().mean()
            else:
                _, labels = torch.max(train_labels, 1)
                accuracy = (labels.squeeze() == argmax.squeeze()).float().mean()

            class_cost_vector.append(class_loss.data)
            cost_vector.append(loss.data)
            acc_vector.append(accuracy.data)
            class_cost_vector.append(class_loss.data)
            domain_cost_vector.append(domain_loss.data)
            cost_vector.append(loss.data)
            acc_vector.append(accuracy.data)
            if i == 0:
                train_score = to_np(class_outputs.squeeze())
                train_pred = to_np(argmax.squeeze())
                train_true = to_np(train_labels.squeeze())
            else:
                class_score = np.concatenate((train_score, to_np(class_outputs.squeeze())), axis=0)
                train_pred = np.concatenate((train_pred, to_np(argmax.squeeze())), axis=0)
                train_true = np.concatenate((train_true, to_np(train_labels.squeeze())), axis=0)
        
        train_accuracy = metrics.accuracy_score(train_true, train_pred)
        train_f1 = metrics.f1_score(train_true, train_pred, average='macro')
        train_precision = metrics.precision_score(train_true, train_pred, average='macro')
        train_recall = metrics.recall_score(train_true, train_pred, average='macro')

        model.eval()
        validate_acc_vector_temp = []
        for i, (validate_data, validate_labels, event_labels) in enumerate(validate_loader):
            validate_text,  validate_mask, validate_labels, event_labels = \
                to_var(validate_data[0]),  to_var(validate_data[1]), \
                to_var(validate_labels), to_var(event_labels)
            validate_outputs, domain_outputs = model(validate_text, validate_mask)
            _, validate_argmax = torch.max(validate_outputs, 1)
            validate_labels=validate_labels.to(torch.long)
            vali_loss = criterion(validate_outputs, validate_labels)
            #domain_loss = criterion(domain_outputs, event_labels)
                #_, labels = torch.max(validate_labels, 1)
            validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
            vali_cost_vector.append( vali_loss.data)
                #validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
            validate_acc_vector_temp.append(validate_accuracy.data)
        validate_acc = np.mean(validate_acc_vector_temp)
        valid_acc_vector.append(validate_acc)
        model.train()

        print('Classification Acc: %.4f'
          % (train_accuracy))
        print("Classification report:\n%s\n"
          % (metrics.classification_report(train_true, train_pred, digits=3)))
        
        # print ('Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, validate loss: %.4f, Train_Acc: %.4f,  Validate_Acc: %.4f.'
        #         % (
        #         epoch + 1, args.num_epochs,  np.mean(cost_vector), np.mean(class_cost_vector), np.mean(vali_cost_vector),
        #             np.mean(acc_vector),   validate_acc, ))


        if validate_acc > best_validate_acc:
            best_validate_acc = validate_acc
            if not os.path.exists(args.output_file):
                os.mkdir(args.output_file)
            best_validate_dir = args.output_file + str(epoch + 1) + '_text.pkl'
            torch.save(model.state_dict(), best_validate_dir)


    duration = time.time() - start_time
    #print ('Epoch: %d, Mean_Cost: %.4f, Duration: %.4f, Mean_Train_Acc: %.4f, Mean_Test_Acc: %.4f'
           #% (epoch + 1, np.mean(cost_vector), duration, np.mean(acc_vector), np.mean(test_acc_vector)))
#best_validate_dir = args.output_file + 'baseline_text_weibo_GPU2_out.' + str(20) + '.pkl'

    # Test the Model
    print('testing model')
    model = CNN_Fusion(args, W)
    model.load_state_dict(torch.load(best_validate_dir))
    #    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    test_score = []
    test_pred = []
    test_true = []
    for i, (test_data, test_labels, event_labels) in enumerate(test_loader):
        test_text,  test_mask, test_labels = to_var(
            test_data[0]), to_var(test_data[1]), to_var(test_labels)
        test_outputs, _= model(test_text, test_mask)
        _, test_argmax = torch.max(test_outputs, 1)
        if i == 0:
            test_score = to_np(test_outputs.squeeze())
            test_pred = to_np(test_argmax.squeeze())
            test_true = to_np(test_labels.squeeze())
        else:
            test_score = np.concatenate((test_score, to_np(test_outputs.squeeze())), axis=0)
            test_pred = np.concatenate((test_pred, to_np(test_argmax.squeeze())), axis=0)
            test_true = np.concatenate((test_true, to_np(test_labels.squeeze())), axis=0)

    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
    test_precision = metrics.precision_score(test_true, test_pred, average='macro')
    test_recall = metrics.recall_score(test_true, test_pred, average='macro')

    test_score_convert = [x[1] for x in test_score]
    test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')
   
    test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)

    print("Classification Acc: %.4f, AUC-ROC: %.4f"
          % (test_accuracy, test_aucroc))
    print("Classification report:\n%s\n"
          % (metrics.classification_report(test_true, test_pred, digits=3)))
    print("Classification confusion matrix:\n%s\n"
          % (test_confusion_matrix))

    print('Saving results')

