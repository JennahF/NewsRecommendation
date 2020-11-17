from collections import Counter
from tqdm import tqdm
import re
import pickle
from utils.dict import Dictionary
import os
import random
import config
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Data preprocess.')
parser.add_argument(
    '-model_name',
    default="NRMS"
)

params = parser.parse_args()

def title_to_max_len(title_vec):
    pad = 48 - len(title_vec)
    if pad != 0:
        title_vec = title_vec + [0] * pad
    return title_vec

def browsed_to_max_num(his_vec, topic_vec, subtopic_vec, abstract_vec):
    dis = config.NRMSconfig.browsed_max_num - len(his_vec)
    if dis > 0:
        for i in range(dis):
            his_vec.append([0] * config.NRMSconfig.title_max_words_num)
            topic_vec.append([0])
            subtopic_vec.append([0])
            abstract_vec.append([0] * config.NAMLconfig.abstract_max_len)
    elif dis < 0:
            his_vec = his_vec[0:config.NRMSconfig.browsed_max_num]
            topic_vec = topic_vec[0:config.NRMSconfig.browsed_max_num]
            subtopic_vec = subtopic_vec[0:config.NRMSconfig.browsed_max_num]
            abstract_vec = abstract_vec[0:config.NRMSconfig.browsed_max_num]
    return his_vec, topic_vec, subtopic_vec, abstract_vec

def candidate_to_max_num(candi_topic, candi_subtopic, candidate, candi_abstract, label):
    dis = config.NRMSconfig.candidate_max_num - len(candi_topic)
    if dis > 0:
        for i in range(dis):
            candi_topic.append([0])
            candi_subtopic.append([0])
            candidate.append([0] * config.NRMSconfig.title_max_words_num)
            candi_abstract.append([0] * config.NAMLconfig.abstract_max_len)
        label = label +  [-1] * (dis)
    elif dis < 0:
        candi_topic = candi_topic[0:config.NRMSconfig.candidate_max_num]
        candi_subtopic = candi_subtopic[0:config.NRMSconfig.candidate_max_num]
        candidate = candidate[0:config.NRMSconfig.candidate_max_num]
        candi_abstract = candi_abstract[0:config.NRMSconfig.candidate_max_num]
        label = label[0:config.NRMSconfig.candidate_max_num]
    return candi_topic, candi_subtopic, candidate, candi_abstract, label

def abstract_to_max_len(abstract):
    dis = config.NAMLconfig.abstract_max_len - len(abstract)
    if dis > 0:
        abstract = abstract + [0] * dis
    elif dis < 0:
        abstract = abstract[0:config.NAMLconfig.abstract_max_len]
    return abstract


def remove_sig(str:str):
    '''remove_sig, remove signals from the input string
    Args:
        str: the input string

    Returns:
        A string without signals like .'", etc
    '''
    return re.sub("[+\.\!\/<>“”''"
                  ",$?\-%^*():+\"\']+|[+——！，。？、~#￥%……&*（）]+", "", str.strip())


def build_news_dict(dirs):
    # f = ['behaviors.tsv', 'news.tsv']
    f = 'news.tsv'
    files = [dir + f for dir in dirs]
    title_dict = Counter()
    topic_dict = Counter()
    subtopic_dict = Counter()
    
    print("building news dictionary...")
    for file in files:
        print(file)
        with open(file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                lines[i] = lines[i].rstrip('\n').split('\t')
                # print(lines[i])
                topic_dict[remove_sig(lines[i][1])] += 1
                subtopic_dict[remove_sig(lines[i][2])] += 1
                title = lines[i][3].split()
                for word in title:
                    title_dict[remove_sig(word)] += 1
                abstract = lines[i][4].split()
                for word in abstract:
                    title_dict[remove_sig(word)] += 1
    
    f = 'behaviors.tsv'
    files = [dir + f for dir in dirs]
    behavior_dict = Counter()
    print("building user behaviour dictionary...")
    for file in files:
        print(file)
        with open(file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip('\n').split('\t')
                behavior_dict[remove_sig(line[1])] += 1

    # print(news_dict)
    print("building news dictionary finished!")
    return topic_dict, subtopic_dict, title_dict, behavior_dict

def vectorize(dir, topic_dic, subtopic_dic, title_dic, vec_cache):
    print("start vectorizing...")
    file = dir + 'news.tsv'
    news_vec = {}
    title_max_len = 0
    abstract_max_len = 0
    with open(file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            temp_vec = {}
            lines[i] = lines[i].rstrip('\n').split('\t')
            # print(lines[i])
            # temp_vec['newsID'] = dictionary.getId(remove_sig(lines[i][0]))
            temp_vec['topic'] = topic_dic.getId(remove_sig(lines[i][1]))
            temp_vec['sub-topic'] = subtopic_dic.getId(remove_sig(lines[i][2]))
            temp_vec['title'] = [title_dic.getId(remove_sig(word)) for word in lines[i][3].split()]
            temp_vec['abstract'] = [title_dic.getId(remove_sig(word)) for word in lines[i][4].split()]
            if title_max_len < len(temp_vec['title']):
                title_max_len = len(temp_vec['title'])
            if abstract_max_len < len(temp_vec['abstract']):
                abstract_max_len = len(temp_vec['abstract'])

            news_vec[remove_sig(lines[i][0])] = temp_vec
    with open(vec_cache, 'wb') as f:
        pickle.dump(news_vec, f)
    print("vectorizing finished!")
    print('max title len = ', title_max_len)
    print('max abstract len = ', abstract_max_len)
    return title_max_len

def ToTrainData(behav_vec, user_dic, news):
    UID = 0
    HTY = 1
    POS = 2
    NEG = 3
    U = []
    T = []
    ST = []
    H = []
    A = []
    CT = []
    CST = []
    C = []
    CA = []
    max_browsed = 0
    for i in range(len(behav_vec)):
        if i%10000 == 0:
            print(i, len(behav_vec))
        uid = user_dic.getId(behav_vec[i][UID])
        history = behav_vec[i][HTY]
        his_vec = []
        topic_vec = []
        subtopic_vec = []
        abstract_vec = []
        for j in range(len(history)):
            # print(history)
            title_vec = title_to_max_len(news[history[j]]['title'])
            his_vec.append(title_vec)
            topic_vec.append([news[history[j]]['topic']])
            subtopic_vec.append([news[history[j]]['sub-topic']])
            abstract_vec.append(abstract_to_max_len(news[history[j]]['abstract']))

        if len(his_vec) > max_browsed:
            max_browsed = len(his_vec)

        his_vec, topic_vec, subtopic_vec, abstract_vec = browsed_to_max_num(his_vec, topic_vec, subtopic_vec, abstract_vec)

        pos_sample = behav_vec[i][POS]
        neg_sample = behav_vec[i][NEG]
        for j in range(len(pos_sample)):
            pos = pos_sample[j]
            neg = random.sample(neg_sample, config.NRMSconfig.neg_sample_num)
            candidate = []
            candi_topic = []
            candi_subtopic = []
            candi_abstract = []

            candidate.append(title_to_max_len(news[pos]['title']))
            candi_topic.append(news[pos]['topic'])
            candi_subtopic.append(news[pos]['sub-topic'])
            candi_abstract.append(abstract_to_max_len(news[pos]['abstract']))

            for n in neg:
                candidate.append(title_to_max_len(news[n]['title']))
                candi_topic.append(news[n]['topic'])
                candi_subtopic.append(news[n]['sub-topic'])
                candi_abstract.append(abstract_to_max_len(news[n]['abstract']))

            U.append([uid])
            T.append(topic_vec)
            ST.append(subtopic_vec)
            H.append(his_vec)
            A.append(abstract_vec)
            CT.append(candi_topic)
            CST.append(candi_subtopic)
            C.append(candidate)
            CA.append(candi_abstract)
    print('max_browsed_train = ', max_browsed)
    return U, T, ST, H, A, CT, CST, C, CA

def ToTestData(behav_vec, user_dic, news):
    '''
    [
        [
            uid, history, candidate[news1], label[0/1]
        ]
    ]
    '''
    UID = 0
    HTY = 1
    POS = 2
    NEG = 3
    U = []
    T = []
    ST = []
    H = []
    A = []
    CT = []
    CST = []
    C = []
    CA = []
    L = []
    I = []
    max_browsed = 0
    max_candidate = 0
    cur_index = 0
    for i in range(len(behav_vec)):
        if i % 10000 == 0:
            print(i, len(behav_vec))
        uid = user_dic.getId(behav_vec[i][UID])
        history = behav_vec[i][HTY]
        his_vec = []
        topic_vec = []
        subtopic_vec = []
        abstract_vec = []
        for j in range(len(history)):
            his_vec.append(title_to_max_len(news[history[j]]['title']))
            topic_vec.append([news[history[j]]['topic']])
            subtopic_vec.append([news[history[j]]['sub-topic']])
            b = abstract_to_max_len(news[history[j]]['abstract'])
            abstract_vec.append(b)
            # print('abstract: ', len(b))

        if len(his_vec) > max_browsed:
            max_browsed = len(his_vec)

        his_vec, topic_vec, subtopic_vec, abstract_vec = browsed_to_max_num(his_vec, topic_vec, subtopic_vec, abstract_vec)

        pos_sample = behav_vec[i][POS]
        pos_label = [1] * len(pos_sample)
        neg_sample = behav_vec[i][NEG]
        neg_label = [0] * len(neg_sample)
        label = pos_label + neg_label

        if len(label) > max_candidate:
            max_candidate = len(label)

        candidate = []
        candi_topic = []
        candi_subtopic = []
        candi_abstract = []
        start = cur_index
        end = cur_index
        for j in range(len(pos_sample)):
            candidate.append(title_to_max_len(news[pos_sample[j]]['title']))
            candi_topic.append([news[pos_sample[j]]['topic']])
            candi_subtopic.append([news[pos_sample[j]]['sub-topic']])
            candi_abstract.append(abstract_to_max_len(news[pos_sample[j]]['abstract']))

        for j in range(len(neg_sample)):
            candidate.append(title_to_max_len(news[neg_sample[j]]['title']))
            candi_topic.append([news[neg_sample[j]]['topic']])
            candi_subtopic.append([news[neg_sample[j]]['sub-topic']])
            candi_abstract.append(abstract_to_max_len(news[neg_sample[j]]['abstract']))

        candi_topic, candi_subtopic, candidate, candi_abstract, label = candidate_to_max_num(candi_topic, candi_subtopic, candidate, candi_abstract, label)
        
        end += len(candi_topic)
        cur_index += len(candi_topic)

        U.append([uid])
        T.append(topic_vec)
        ST.append(subtopic_vec)
        H.append(his_vec)
        CT.append(candi_topic)
        CST.append(candi_subtopic)
        C.append(candidate)
        L[len(L):len(L)] = label
        I.append([start,end])
        A.append(abstract_vec)
        CA.append(candi_abstract)
    print('max_browsed_test = ', max_browsed)
    print('max_candidate_test = ', max_candidate)
    return U, T, ST, H, A, CT, CST, C, CA, L, I

def vectorize_behaviors(dir, dic, news, behavior_cache, test):
    '''
        [
            [userId, t, history, pos_sample, neg_sample]
        ]
    '''
    print("start vectorizing...")
    file = dir + 'behaviors.tsv'
    behav_vec = []
    with open(file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            data = lines[i].rstrip('\n').split('\t')
            userId = data[1]
            history = data[3].split()
            impression = data[4].split()
            pos_sample = []
            neg_sample = []
            for i in range(len(impression)):
                imp = impression[i].split('-')
                if imp[1] == '1':
                    pos_sample.append(imp[0])
                else:
                    neg_sample.append(imp[0])
            behav_vec.append([userId, history, pos_sample, neg_sample])
    # with open(behavior_cache, 'wb') as f: 
    #     pickle.dump(behav_vec, f) 

    if test:
        U, T, ST, H, A, CT, CST, C, CA, L, I = ToTestData(behav_vec, dic, news)
        with open(behavior_cache, 'wb') as f: 
            pickle.dump([U, T, ST, H, [], CT, CST, C, [], L, I], f) 
        print('vectorizing finished!')

    else:
        U, T, ST, H, A, CT, CST, C, CA = ToTrainData(behav_vec, dic, news)
        with open(behavior_cache, 'wb') as f: 
            pickle.dump([U, T, ST, H, A, CT, CST, C, CA], f) 
        print('vectorizing finished!')
                


def main():
    path = './dataset/'
    dirs = ['MINDsmall_train/', 'MINDsmall_dev/']
    dirs = [path + dir for dir in dirs]
    
    news_dic_cache = './temp/news_dictionary.pickle'
    if os.path.exists(news_dic_cache):
        print("news dictionary found!\n")
        with open(news_dic_cache, 'rb') as f:
            dictionary = pickle.load(f)
            topic_dic, subtopic_dic, title_dic, behav_dic = dictionary[0], dictionary[1], dictionary[2], dictionary[3]
            print('topic num:', topic_dic.vocab_len)
            print('sub-topic num:', subtopic_dic.vocab_len)
            print('behav num:', behav_dic.vocab_len)
    else:
        print("news dictionary not found!")
        topic_dic, subtopic_dic, title_dic, behav_dic = build_news_dict(dirs)
        topic_dic, _ = zip(*topic_dic.most_common())
        subtopic_dic, _ = zip(*subtopic_dic.most_common())
        title_dic, _ = zip(*title_dic.most_common())
        behav_dic, _ = zip(*behav_dic.most_common())
        topic_word2id = {token: i+1 for i, token in enumerate(topic_dic)}
        subtopic_word2id = {token: i+1 for i, token in enumerate(subtopic_dic)}
        title_word2id = {token: i+1 for i, token in enumerate(title_dic)}
        behav_word2id = {token: i+1 for i, token in enumerate(behav_dic)}
        
        topic_dic = Dictionary(topic_word2id)
        subtopic_dic = Dictionary(subtopic_word2id)
        title_dic = Dictionary(title_word2id)
        behav_dic = Dictionary(behav_word2id)

        with open(news_dic_cache, 'wb') as f:
            pickle.dump([topic_dic, subtopic_dic, title_dic, behav_dic], f)
        print(title_dic.vocab_len)

    #新闻文本 -> id序列
    vec_cache = ['./temp/train_news_vec.pickle', './temp/test_news_vec.pickle']
    title_max_len = 0
    for i in range(2):
        if os.path.exists(vec_cache[i]):
            print(vec_cache[i][7:-16], "news vector found!\n")
        else:
            print(vec_cache[i][7:-16], "news vector not found!")
            temp_max_len = vectorize(dirs[i], topic_dic, subtopic_dic, title_dic, vec_cache[i])
            if title_max_len < temp_max_len:
                title_max_len = temp_max_len
    # print("title max len = ", title_max_len)

    
    #用户行为向量化
    behaviors_cache = ['./temp/train_behavior.pickle', './temp/test_behavior.pickle']
    for i in range(2):
        if os.path.exists(behaviors_cache[i]):
            print(behaviors_cache[i][7:-16], "behavior found!\n")
        else:
            print(behaviors_cache[i][7:-16], "behavior not found!")
            with open(vec_cache[i], 'rb') as f:
                news = pickle.load(f)
            vectorize_behaviors(dirs[i], behav_dic, news, behaviors_cache[i], i)

if __name__ == '__main__':
    main()

    # with open('./temp/train_behavior.pickle', 'rb') as f:
    #     train_data = pickle.load(f)
    # print(len(train_data))
    # print(len(train_data[0]), len(train_data[0][0]))
    # print(len(train_data[1]), len(train_data[1][0]))
    # print(len(train_data[2]), len(train_data[2][0]))
    # # print(train_data[sample_num][browsed_or_not_clicked][i]['sub-category'])
