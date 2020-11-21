import torch
import argparse
import config
from utils.DataLoader import myCandidateLoader, myNewsLoader, myTestDataLoader, myUserLoader
# from utils import DataLoader
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
    '-testdata_news',
    default='./temp/test_news_vec.pickle'
)

parser.add_argument(
    '-testdata_history',
    default='./temp/test_behavior_history.pickle'
)

parser.add_argument(
    '-testdata_candidate',
    default='./temp/test_behavior_candidate.pickle'
)

parser.add_argument(
    '-batch_size',
    type=int,
    default=2
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



def testModel(model):
    '''testloader:
        [
            [uid, history, candidate[[news1],[news2],...], label[0,1,...]]
        ]
    '''
    model.eval()

    '''
        [
            [u],...,[u] #impression size*sample num
        ]
    '''
    '''
    preds = []
    true = []
    for i in range(test_loader.batch_num):
        if i == 100:
            break

        testdata = test_loader[i]
        history, candidate = testdata['history'], testdata['candidate']
        print(history.size())

        # cpu_history, gpu1_history, gpu2_history, gpu3_history, gpu4_history = history.split(5, dim=0)
        # cpu_candidate, gpu1_candidate, gpu2_candidate, gpu3_candidate, gpu4_candidate = candidate.split(5, dim=0)
        click_prob = model([history, candidate])

        # gpu_history = torch.cat([gpu1_history, gpu2_history, gpu3_history, gpu4_history])
        # gpu_candidate = torch.cat([gpu1_candidate, gpu2_candidate, gpu3_candidate, gpu4_candidate])
        # gpu_click_prob = model([gpu_history.to(device), gpu_candidate.to(device)])
        # click_prob = torch.cat((cpu_click_prob, gpu_click_probs))
        with torch.no_grad():
            if i == 0:
                # print(click_prob.size())
                preds = click_prob.transpose(0,1).squeeze(0)
            else:
                preds = torch.cat((preds, click_prob.transpose(0,1).squeeze(0)))
        
        # del gpu_history
        # del gpu_candidate
        # del cpu_history
        # del cpu_candidate
        # del gpu1_candidate
        # del gpu2_candidate
        # del gpu3_candidate
        # del gpu4_candidate
        # del gpu1_history
        # del gpu2_history
        # del gpu3_history
        # del gpu4_history
        # gc.collect()
    '''
    print('start testing...')

    with open(params.testdata_news, 'rb') as v:
        news_vec = pickle.load(v)
    news_loader = myNewsLoader(news_vec, params.model_name, params.batch_size)
    encoded_news = []
    news_id = []
    with torch.no_grad():
        for i in range(news_loader.batch_num):
            #[1, bs], [bs*1, bs*1, bs*title_words_num, bs*abstract_words_num]
            newsid, news_vector = news_loader[i]
            #[bs * embedding_dim]
            news = model.module.get_news_encode(torch.Tensor(news_vector).cuda())

            if i == 0:
                encoded_news = news
                news_id = newsid
            else:
                #[news_num * embedding_size]
                encoded_news = torch.cat(encoded_news, news)
                news_id = news_id + newsid

    encoded_news = {news_id[i]: encoded_news[i] for i in range(len(news_id))}
    
    with open(params.testdata_history, 'rb') as v:
        test_vec = pickle.load(v)

    user_loader = myUserLoader(encoded_news, test_vec[0], test_vec[1], params.batch_size, params.model_name)
    encoded_user = []
    with torch.no_grad():
        for i in range(user_loader.batch_num):
            #[bs * browsed_num * embedding_dim]
            if params.model_name != 'LSTUR':
                history = user_loader[i]
                user = model.moudel.get_user_encode(torch.Tensor(history).cuda())
            else:
                history, userid = user_loader[i]
                user = model.module.get_user_encode(torch.Tensor(history).cuda(), torch.Tensor(userid).cuda())

            if i == 0:
                encoded_user = user
            else:
                #[sample_num * embedding_size]
                encoded_user = torch.cat(encoded_user, user)

    with open(params.testdata_candidate, 'rb') as v:
            candi_vec = pickle.load(v)
    candidate_loader = myCandidateLoader(candi_vec[0], encoded_user, encoded_news, params.batch_size)
    probs = []
    with torch.no_grad():
        for i in range(candidate_loader.batch_num):
            #[bs * candidate_max_num * embedding_size]
            #[bs * embedding_size]
            candidate, user = candidate_loader[i]
            #[bs * candidate_max_num]
            prob = torch.bmm(candidate, user.unsqueeze(1).transpose(1,2)).squeeze(2)

            if i == 0:
                probs = torch.reshape(prob, (1, prob.size()[0]*prob.size()[1]))
            else:
                probs = torch.cat(probs, torch.reshape(prob, (1, prob.size()[0]*prob.size()[1])))
        
    true = candi_vec[1]
        
    auc, mrr, ndcg5, ndcg10 = evaluate(impression_index, probs.tolist(), true.tolist())
    print('auc:', auc)
    print('mrr:', mrr)
    print('ndcg5:', ndcg5)
    print('ndcg10:', ndcg10)

def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

    
    # myTestLoader = DataLoader.myTestDataLoader(test_vec, params.batch_size)

    if params.model_name == 'NRMS':
        model = NRMS(config.NRMSconfig)
    elif params.model_name == 'LSTUR':
        model = LSTUR(config.LSTURconfig, params.ini)
    elif params.model_name == 'NAML':
        model = NAML()

    model.load_state_dict(torch.load(FindModelPath()), False)

    model = nn.DataParallel(model, device_ids=[1,2,3])
    model.cuda()
    # model = model.to(device)()

    # model = model.to(device)()
    # model = torch.nn.DataParallel(model)
    # checkpoint = 
    # model.load_state_dict(torch.load(FindModelPath()), False)

    torch.cuda.empty_cache()

    testModel(model)

if __name__ == '__main__':
    main()