import argparse
import pickle
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import DataLoader
import config
from utils import DataSet
import numpy as np
from sklearn import metrics
import torch.nn.functional as F
from model.model import NRMS, LSTUR, NAML
import torch.nn as nn
import os

parser = argparse.ArgumentParser(description='Train NRMS model.')

parser.add_argument(
    '-traindata',
    default='./temp/train_behavior.pickle'
)

parser.add_argument(
    '-testdata',
    default='./temp/test_behavior.pickle'
)

parser.add_argument(
    '-dict',
    default='./temp/news_dictionary.pickle'
)

parser.add_argument(
    '-newstrain',
    default='./temp/train_news_vec.pickle'
)

parser.add_argument(
    '-newstest',
    default='./temp/test_news_vec.pickle'
)

parser.add_argument(
    '-model_path',
    default='./model'
)

parser.add_argument(
    '-batch_size',
    type=int,
    default=64
)

parser.add_argument(
    '-category_embedding_dim',
    type=int,
    default=100
)

parser.add_argument(
    '-model_name',
    default='NRMS'
)

parser.add_argument(
    '-lr',
    type=float,
    default=0.001
)

parser.add_argument(
    '-dropout',
    type=float,
    default=0.0001
)

parser.add_argument(
    '-epoch',
    type = int,
    default=2
)

parser.add_argument(
    '-load_model',
    type = int,
    default=0
)

parser.add_argument(
    '-model_num',
    type=int,
    default=0
)

parser.add_argument(
    '-ini',
    type=int,
    default=1
)

global modelnum 
modelnum = -1

def FindModelPath():
    global modelnum
    if params.model_num != 0:
        path = params.model_path + '/' + params.model_name + '/model_' + str(params.model_num) + '.pkl'
        modelnum = params.model_num+1
    else:
        names =  os.listdir(params.model_path + '/' + params.model_name)
        path = params.model_path + '/' + params.model_name + '/' + names[-1]
        modelnum = int(names[-1][-5])+1
    return path

params = parser.parse_args()
# print(params)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def evaluate(impression_index, preds, label):
    auc_scores = []
    mrr_scores = []
    ndcg5_scores = []
    ndcg10_scores = []
    for i in range(len(impression_index)):
        s = impression_index[i][0]
        e = impression_index[i][1]
        auc = metrics.roc_auc_score(preds[s:e], label[s:e])
        mrr_scores.append(mrr)
        if e - s > 4:
            ndcg5 = ndcg_score(preds[s:e], label[s:e], 5)
            ndcg5_scores.append(ndcg5)
            if e - s > 9:
                ndcg10 = ndcg_score(preds[s:e], label[s:e], 10)
                ndcg10_scores.append(ndcg10)
    return np.mean(auc_scores), np.mean(mrr_scores), np.mean(ndcg5_scores), np.mean(ndcg10_scores)

def testModel(model, test_loader, model_name):
    '''testloader:
        [
            [uid, history, candidate[[news1],[news2],...], label[0,1,...]]
        ]
    '''
    model.eval()

    #news and user encoding
    test_output = []
    test_label = []
    '''
        [
            [u],...,[u] #impression size*sample num
        ]
    '''
    preds = []
    true = []
    for i in range(test_loader.batch_num):
        if i == 100:
            break
        testdata = test_loader[i]
        uid, topic, subtopic, history, candi_topic, candi_subtopic, candidate = \
                        minibatch['userId'], minibatch['topic'], minibatch['sub-topic'], minibatch['history'], \
                        minibatch['candi-topic'], minibatch['candi-subtopic'], minibatch['candidate']
        if params.model_name == 'NRMS':
            click_prob = model([history.cuda(), candidate.cuda()])
        elif params.model_name == 'LSTUR':
            click_prob = model([uid.cuda(1), topic.cuda(1), subtopic.cuda(1), history.cuda(1), candi_topic.cuda(1), candi_subtopic.cuda(1), candidate.cuda(1)])
        
        if i == 0:
            print(click_prob.size())
            preds = click_prob.transpose(0,1).squeeze(0)
        else:
            preds = torch.cat((preds, click_prob.transpose(0,1).squeeze(0)))
        
    impression_index = test_loader.getImpressionIndex()
    true = test_loader.getLabels()
        
    auc, mrr, ndcg5, ndcg10 = evaluate(impression_index, preds.tolist(), true.tolist())
    print('auc:', auc)
    print('mrr:', mrr)
    print('ndcg5:', ndcg5)
    print('ndcg10:', ndcg10)

def NegSampleloss(probs):
    return torch.stack([stm[0] for stm in -F.log_softmax(probs, dim=1)]).mean()

def Resize(batched_data):
    '''
    input: sample num*dim*batch_size
    output: batch_size*dim*sample num
    '''
    a = batched_data.transpose(0,2)
    b = a.transpose(1,2)
    return b

def BigTensor(minibatch):
    for i in range(len(minibatch)):
        for j in range(len(minibatch[i])):
            minibatch[i][j] = minibatch[i][j].tolist()
    return torch.Tensor(minibatch)

# def trainModel(model, train_loader, test_loader, optimizer):
def trainModel(model, train_loader, optimizer):
    def trainEpoch(epoch):
        global modelnum
        epoch_loss = []
        for i in range(train_loader.batch_num):
            minibatch = train_loader[i]
            uid, topic, subtopic, history, candi_topic, candi_subtopic, candidate, abstract, candi_abstract = \
                        minibatch['userId'].cuda(1), minibatch['topic'].cuda(1), minibatch['sub-topic'].cuda(1), minibatch['history'].cuda(1), \
                        minibatch['candi-topic'].cuda(1), minibatch['candi-subtopic'].cuda(1), minibatch['candidate'].cuda(1), minibatch['abstract'].cuda(1), minibatch['candi-abstract']

            optimizer.zero_grad()
            if params.model_name == 'NRMS':
                click_prob = model([history, candidate])
                loss = NegSampleloss(click_prob)
            elif params.model_name == 'LSTUR':
                print('h size:', history.size())
                print('topic size:', topic.size())
                click_prob = model([uid, topic, subtopic, history, candi_topic, candi_subtopic, candidate])
                loss = NegSampleloss(click_prob)
            elif params.model_name == 'NAML':
                click_prob = model([topic, subtopic, history, abstract, candi_topic, candi_subtopic, candidate, candi_abstract])
                loss = NegSampleloss(click_prob)
            else:
                pass

            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            if i % 200 == 0:
                print(i)

            if i % 1000 == 0:
                print('Iteration:{0}, train loss:{1}, avg loss:{2}'.format(i+1, loss, np.mean(epoch_loss)))

        torch.save(model.state_dict(), params.model_path + '/' + params.model_name + '/model_%s.pkl' % format(modelnum))
        modelnum += 1
        # acc = testModel(model, test_loader)
        return np.mean(epoch_loss)

    model.train()
    for epoch in range(params.epoch):
        print('.....start training.....')
        train_loss = trainEpoch(epoch)
        print('Epoch %d:\t average loss: %.2f\t ' % (epoch, train_loss))


def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

    # if torch.cuda:
    #     torch.cuda.set_device(device)

    print('Start loading data!')

    with open(params.traindata, 'rb') as t:
        train_vec = pickle.load(t)

    myTrainLoader = DataLoader.myTrainDataLoader(train_vec, params.batch_size)

    print("Data loading finished!", myTrainLoader.batch_num)

    if params.model_name == 'NRMS':
        model = NRMS(config.NRMSconfig)
    elif params.model_name == 'LSTUR':
        model = LSTUR(config.LSTURconfig, params.ini)
    elif params.model_name == 'NAML':
        model = NAML(config.NAMLconfig)
    
    optimizer = Adam(model.parameters(), lr=params.lr, weight_decay=params.dropout)

    print(sum(param.numel() for param in model.parameters()))

    if params.load_model:
        model.load_state_dict(torch.load(FindModelPath()), False)
    
    model = nn.DataParallel(model, device_ids=[1,2,3])
    model = model.cuda(1)

    trainModel(model, myTrainLoader, optimizer)

if __name__ == '__main__':
    main()