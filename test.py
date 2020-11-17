import torch
import argparse
import config
from utils import DataLoader
from model.model import NRMS, LSTUR
import os
import pickle
import torch.nn as nn
import gc

parser = argparse.ArgumentParser(description='Test model.')

parser.add_argument(
    '-model_num',
    default = 0
)

parser.add_argument(
    '-model_name',
    default='NRMS'
)

parser.add_argument(
    '-model_path',
    default='./model'
)

parser.add_argument(
    '-testdata',
    default='./temp/test_behavior.pickle'
)

parser.add_argument(
    '-batch_size',
    type=int,
    default=25
)

parser.add_argument(
    '-ini',
    type=int,
    default=1
)

params = parser.parse_args()
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

def FindModelPath():
    if params.model_num != 0:
        path = params.model_path + '/' + params.model_name + '/model_' + str(params.model_num) + '.pkl'
    else:
        names =  os.listdir(params.model_path + '/' + params.model_name)
        path = params.model_path + '/' + params.model_name + '/' + names[-1]
    return path

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
    for i in range(len(preds)):
        s = impression_index[i][0]
        e = impression_index[i][1]
        while e > s and label[e] == -1:
            e -= 1
        auc = metrics.roc_auc_score(preds[s:e], label[s:e])
        mrr_scores.append(mrr)
        if e - s > 4:
            ndcg5 = ndcg_score(preds[s:e], label[s:e], 5)
            ndcg5_scores.append(ndcg5)
            if e - s > 9:
                ndcg10 = ndcg_score(preds[s:e], label[s:e], 10)
                ndcg10_scores.append(ndcg10)
    return np.mean(auc_scores), np.mean(mrr_scores), np.mean(ndcg5_scores), np.mean(ndcg10_scores)

def testModel(model, test_loader):
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
        history, candidate = testdata['history'], testdata['candidate']
        print(history.size())

        cpu_history, gpu1_history, gpu2_history, gpu3_history, gpu4_history = history.split(5, dim=0)
        cpu_candidate, gpu1_candidate, gpu2_candidate, gpu3_candidate, gpu4_candidate = candidate.split(5, dim=0)
        cpu_click_prob = model([cpu_history, cpu_candidate])

        gpu_history = torch.cat([gpu1_history, gpu2_history, gpu3_history, gpu4_history])
        gpu_candidate = torch.cat([gpu1_candidate, gpu2_candidate, gpu3_candidate, gpu4_candidate])
        gpu_click_prob = model([gpu_history.to(device), gpu_candidate.to(device)])
        click_prob = torch.cat((cpu_click_prob, gpu_click_prob))
        # if i == 0:
        #     print(click_prob.size())
        #     preds = click_prob.transpose(0,1).squeeze(0)
        # else:
        #     preds = torch.cat((preds, click_prob.transpose(0,1).squeeze(0)))
        
        del gpu_history
        del gpu_candidate
        del cpu_history
        del cpu_candidate
        del gpu1_candidate
        del gpu2_candidate
        del gpu3_candidate
        del gpu4_candidate
        del gpu1_history
        del gpu2_history
        del gpu3_history
        del gpu4_history
        gc.collect()
        
    impression_index = test_loader.getImpressionIndex()
    true = test_loader.getLabels()
        
    # auc, mrr, ndcg5, ndcg10 = evaluate(impression_index, preds.tolist(), true.tolist())
    # print('auc:', auc)
    # print('mrr:', mrr)
    # print('ndcg5:', ndcg5)
    # print('ndcg10:', ndcg10)

def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    with open(params.testdata, 'rb') as v:
        test_vec = pickle.load(v)
    myTestLoader = DataLoader.myTestDataLoader(test_vec, params.batch_size)

    print('test data loaded!')

    if params.model_name == 'NRMS':
        model = NRMS(config.NRMSconfig)
    elif params.model_name == 'LSTUR':
        model = LSTUR(config.LSTURconfig, params.ini)
    elif params.model_name == 'NAML':
        model = NAML()

    model.load_state_dict(torch.load(FindModelPath()), False)

    model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.to(device)
    # model = model.to(device)()

    # model = model.to(device)()
    # model = torch.nn.DataParallel(model)
    # checkpoint = 
    # model.load_state_dict(torch.load(FindModelPath()), False)

    torch.cuda.empty_cache()

    testModel(model, myTestLoader)

if __name__ == '__main__':
    main()