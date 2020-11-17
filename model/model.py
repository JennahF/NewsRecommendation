import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from config import LSTURconfig, NRMSconfig, NAMLconfig

device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device3 = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, head_num, embedding_dim, output_dim):
        super(MultiHeadSelfAttention, self).__init__()
        self.head_num = head_num
        self.input_dim = embedding_dim
        self.head_output_dim = output_dim

        self.Q = nn.ParameterList([nn.Parameter(torch.randn(self.input_dim, self.input_dim)) for _ in range(self.head_num)])
        self.V = nn.ParameterList([nn.Parameter(torch.randn(self.head_output_dim, self.input_dim)) for _ in range(self.head_num)])
    
    def forward(self, title_embedding):
        '''
        input: [batch_size, browsed_num, max_word_num, embedding_size]
        output: [batch_size, browsed_num, max_word_num, embedding_size]
        '''
        bs = title_embedding.size()[0]
        browsed_num = title_embedding.size()[1]

        input_dim_num = len(title_embedding.size())

        if input_dim_num == 4:
            #print(title_embedding.size(), self.Q[0].size())
            title_embedding = torch.reshape(title_embedding, (bs*browsed_num, title_embedding.size()[2], title_embedding.size()[3]))
            #print(title_embedding.size(), self.Q[0].size())

        # assert title_embedding.size()[2] == self.input_dim 
        H = []
        for k in range(self.head_num):

            alpha = torch.matmul(title_embedding, self.Q[k])
            alpha = torch.bmm(alpha, title_embedding.transpose(1,2))
            alpha_k = torch.softmax(alpha, dim=2)
            hk = torch.matmul(alpha_k, title_embedding)
            hk = torch.matmul(hk, self.V[k].t())

            if k == 0:
                H = hk
            else:
                H = torch.cat((H, hk), dim=2)
        if input_dim_num == 4:
            H = torch.reshape(H, (bs, browsed_num, H.size()[1], H.size()[2]))

        #print(H.size())
        return H

class AdditiveAttention(nn.Module):
    def __init__(self, query_dim, input_dim):
        super(AdditiveAttention, self).__init__()
        self.query_dim = query_dim
        self.input_dim = input_dim
        self.V = nn.Linear(input_dim, query_dim)
        self.q = nn.Parameter(torch.randn(query_dim))
    
    def forward(self, title_represent):
        '''
        input:
            [
                [
                    [ ],
                    [ ],
                    ...
                ] * batch_size
            ]
        output:
            [
                [ ] * batch_size
            ]
        '''
        #bs * browsed_num * query_dim

        # bs = title_represent.size()[0]
        # browsed_num = title_represent.size()[1]
        # title_represent = torch.reshape(title_represent, (bs*browsed_num, title_represent.size()[2], title_represent.size()[3]))

        a = torch.tanh(self.V(title_represent))

        a = torch.matmul(a, self.q.t())

        if len(title_represent.size()) == 4:
            alpha = F.softmax(a, dim=2)
            r = torch.matmul(alpha.unsqueeze(2), title_represent).squeeze(2)
        else:
            alpha = F.softmax(a, dim=1)
            r = torch.matmul(alpha.unsqueeze(1), title_represent).squeeze(1)
        # r = torch.reshape(r, (bs, browsed_num, r.size()[1], r.size()[2]))
        return r

class NRMS_NewsEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, selfatt_head, query_dim, selfatt_output_dim, dropout):
        super(NRMS_NewsEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.selfatt_head = selfatt_head
        self.query_dim = query_dim
        self.selfatt_output_dim = selfatt_output_dim
        self.dropout = dropout
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.MultiHeadSelfAttention = MultiHeadSelfAttention(self.selfatt_head, self.embedding_dim, self.selfatt_output_dim)
        self.attention = AdditiveAttention(self.query_dim, self.selfatt_output_dim * self.selfatt_head)

    def forward(self, title):
        '''
        input:tensor
            [
                [
                    [x, x, ...(48)] * browsed num or candidate num
                ] * batch_size
            ]
        output:
            [
                [
                    [x, x, ...(16*16=256)] * browsed num or candidate num
                ] * batch_size
            ]
        '''
        ##print(1.1)
        # title_embedding = nn.Dropout(self.dropout)(self.embedding(torch.Tensor(title)))
        title_embedding = self.embedding(title)
        ##print(1.2)
        selfatt_output = nn.Dropout(self.dropout)(self.MultiHeadSelfAttention(title_embedding))
        ##print(1.3)
        u = self.attention(selfatt_output)
        ##print(1.4)
        return u

class NRMS_UserEncoder(nn.Module):
    def __init__(self, embedding_dim, selfatt_head, query_dim, selfatt_output_dim, dropout):
        super(NRMS_UserEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.selfatt_head = selfatt_head
        self.query_dim = query_dim
        self.selfatt_output_dim = selfatt_output_dim
        self.dropout = dropout
        self.MultiHeadSelfAttention = MultiHeadSelfAttention(self.selfatt_head, self.embedding_dim, self.selfatt_output_dim)
        self.attention = AdditiveAttention(self.query_dim, self.selfatt_output_dim * self.selfatt_head)

    def forward(self, news_encode):
        
        sa_op = nn.Dropout(self.dropout)(self.MultiHeadSelfAttention(news_encode))
        u = self.attention(sa_op)
        return u
        
class NRMS(nn.Module):
    def __init__(self, config):
        super(NRMS, self).__init__()
        self.config = config
        self.NewsEncoder = NRMS_NewsEncoder(self.config.vocab_size, self.config.word_embedding_dim, self.config.selfatt_head, self.config.query_dim, self.config.selfatt_output_dim, self.config.dropout)
        self.UserEncoder = NRMS_UserEncoder(self.config.word_embedding_dim, self.config.selfatt_head, self.config.query_dim, self.config.selfatt_output_dim, self.config.dropout)
    
    def forward(self, data):
        '''
        input:
            [
                [
                    userId, history, candidate(K+1)
                ] * batch_size
            ]
        output:
            [
                [
                    [ ]*K+1
                ] * batch_size
            ]
        '''
        history, candidate = data[0], data[1]
        bs = history.size()[0]
        browsed_num = history.size()[1]
        candidate_num = candidate.size()[1]

        history = torch.reshape(history, (bs*browsed_num, history.size()[2]))
        news_encode = self.NewsEncoder(history)
        #batch_size * browsed_num * |r|
        news_encode = torch.reshape(news_encode, (bs, browsed_num, news_encode.size()[1]))
        #batch_size * |u|
        user_encode = self.UserEncoder(news_encode)

        candidate = torch.reshape(candidate, (bs*candidate_num, candidate.size()[2]))
        candidate_encode = self.NewsEncoder(candidate)
        #batch_size * |K+1| * |r|
        candidate_encode = torch.reshape(candidate_encode, (bs, candidate_num, candidate_encode.size()[1]))

        click_prob = torch.sum(candidate_encode.mul(user_encode.unsqueeze(1)), dim=2)
        return click_prob

    def get_model_output(self, history):
        '''
        input:
            [uid, history, candidate[[news1],[news2],...], label[0,1,...]]
        output:
            [
                [u],...,[u] #impression size*sample num
            ]
        '''
        batch_size = len(history)
        #print('history: ', history.size())
        history_encode = self.NewsEncoder(history)
        #print('history_encode: ', history_encode.size())
        user_encode = self.UserEncoder(history_encode)
        #print('user_encode: ', user_encode.size())
        return user_encode
    
    def get_click_probs(self, u, candidate):
        # return torch.sum(torch.Tensor(u.to(device0)).mul(torch.Tensor(candidate).to(device0)), dim=1).numpy().tolist()
        return torch.sum(torch.Tensor(u).mul(torch.Tensor(candidate)), dim=1).numpy().tolist()
    

class LSTUR_NewsEncoder(nn.Module):
    def __init__(self, config : LSTURconfig):
        super(LSTUR_NewsEncoder, self).__init__()
        self.config = config
        self.word_embedding = nn.Embedding(self.config.vocab_size, self.config.word_embedding_dim, padding_idx=0)
        self.topic_embedding = nn.Embedding(self.config.topic_num, self.config.topic_embedding_dim, padding_idx=0)
        self.subtopic_embedding = nn.Embedding(self.config.topic_num, config.subtopic_embedding_dim, padding_idx=0)
        self.CNN = nn.Conv2d(1, self.config.filter_num, (self.config.windows_size, self.config.word_embedding_dim), padding=(1,0))
        self.attention = AdditiveAttention(self.config.query_dim, self.config.filter_num)
        self.dropout = nn.Dropout(self.config.dropout)
        self.ReLU = nn.ReLU(inplace=True)

    
    def forward(self, data):
        '''
        input:
        [
            topic, (batch_size * browsed_num * 1)
            sub-topic, (batch_size * browsed_num * 1)
            history (batch_size * browsed_num * max_word_num)
        ]
        '''
        topic = data[0]
        subtopic = data[1]
        history = data[2]

        print(0.1)
        #batch_size * browsed_num * topic_embedding_dim
        topic_embed = self.topic_embedding(topic)
        subtopic_embed = self.subtopic_embedding(subtopic)

        print(0.2)
        print('CNN:', self.CNN)
        print('history size:', history.size())
        print('word_embedding:', self.word_embedding)
        history_embed = self.word_embedding(history)
        print('history embed size:', history_embed.size())

        bs = history_embed.size()[0]
        browsed_num = history_embed.size()[1]
        #bs * 1 * title_max_num * word_embedding_num
        history_embed = torch.reshape(history_embed, (bs*browsed_num, history_embed.size()[2], history_embed.size()[3])).unsqueeze(1)
        print('history embed size:', history_embed.size())
        history_context_represent = self.CNN(history_embed)

        print(0.3)

        history_context_represent = torch.reshape(history_context_represent, (bs, browsed_num, history_context_represent.size()[1], history_context_represent.shape()[2]))
        #batch_size * browsed_num * |et|
        e = self.dropout(self.attention(history_context_represent))
        e = torch.cat((subtopic_embed, e), dim=2)
        e = torch.cat((topic_embed, e), dim=2)

        return e

class LSTUR_UserEncoder(nn.Module):
    def __init__(self, config : LSTURconfig, ini):
        super(LSTUR_UserEncoder, self).__init__()
        self.ini = ini

        assert (config.filter_num+config.topic_embedding_dim+config.subtopic_embedding_dim) % 2 == 0

        if self.ini:
            self.gru_input_dim = config.filter_num+config.topic_embedding_dim+config.subtopic_embedding_dim
        else:
            self.gru_input_dim = (config.filter_num+config.topic_embedding_dim+config.subtopic_embedding_dim)/2
        
        self.gru = nn.GRU(self.gru_input_dim, self.gru_input_dim, batch_first=True)
        self.user_embed = nn.Embedding(config.user_embedding_dim, self.gru_input_dim)
    
    def forward(self, news_represent, userId):
        print(1.1)
        userId_embed = F.dropout(self.user_embed(userId), p=self.config.mask_prob, training=True)
        print(1.2)
        #hn: batch_size * gru_input_dim
        #userId_embed:batch_size * gru_input_dim
        if self.ini:
            _, hn = self.gru(news_represent, userId_embed)
            print(1.3)
            return hn
        else:
            _, hn = self.gru(news_represent)
            return torch.cat((hn, userId_embed), dim=1)
        

class LSTUR(nn.Module):
    def __init__(self, config:LSTURconfig , ini):
        super(LSTUR, self).__init__()
        self.NewsEncoder = LSTUR_NewsEncoder(config)
        self.UserEncoder = LSTUR_UserEncoder(config, ini)
    
    def forward(self, data):
        '''
        input:
            [
                [
                    userId, history, candidate(K+1)
                ] * batch_size
            ]
        output:
            [
                [
                    [ ]*K+1
                ] * batch_size
            ]
        '''
        userId = data[0] #batch_size * 1
        topic = data[1] #bs * browsed_num * 1
        subtopic = data[2] #bs * browsed_num * 1
        history = data[3] #bs * browsed_num * title_max_words
        candi_topic = data[4] #bs * |K+1| * 1
        candi_subtopic = data[5] #bs * |K+1| * 1
        candidate = data[6] #bs * |K+1| * title_max_words

        print('history size:', history.size())
        print('topic size:', topic.size())

        print(0)
        news_encode = self.NewsEncoder([topic, subtopic, history])
        print(1)
        user_encode = self.UserEncoder(news_encode, userId) #batch_size * gru_output_dim
        print(2)
        #batch_size * |K+1| * |et|
        candidate_encode = self.NewsEncoder([candi_topic, candi_subtopic, candidate])
        user_encode = user.unsqueeze(1).transpose(1,2) #bs * gru_output_dim * 1
        click_prob = torch.matmul(candidate_encode, user_encode).unsqueeze(2)

        return click_prob



class NAML_NewsEncoder(nn.Module):
    def __init__(self, config : NAMLconfig, pretrained_embedding=None):
        super(NewsEncoder, self).__init__()
        self.config = config
        if pretrained_embedding is None:
            self.word_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(pretrained_embedding,freeze=False)
        self.topic_embedding = nn.Embedding(config.topic_num, config.topic_embedding_dim, padding_idx=0)
        self.subtopic_embedding = nn.Embedding(config.subtopic_num, config.topic_embedding_dim, padding_idx=0)
        
        self.title_abstract_cnn = nn.Conv2d(config.browsed_max_num, config.filter_num, (config.window_size, config.embedding_dim), padding=(1, 0))
        self.abstract_cnn = nn.Conv2d(1, config.filter_num, (config.window_size, config.embedding_dim), padding=(1, 0))
        self.title_attention = Attention(config.query_dim, config.filter_num)
        self.abstract_attention = Attention(config.query_dim, config.filter_num)

        self.topic_dense = nn.Linear(config.topic_embedding_dim, config.filter_num)
        self.subtopic_dense = nn.Linear(config.topic_embedding_dim, config.filter_num)

        self.view_attention = Attention(config.query_dim, config.filter_num)
    
    def forward(self, topic, subtopic, title, abstract):

        #title
        # bs * browsed_max_num * title_words_num * embedding_dim
        title_embedded = F.dropout(self.word_embedding(title), p=self.config.dropout)

        # bs * browsed_max_num * title_words_num * filter_num
        title_cnn = self.title_abstract_cnn(title_embedded)
        title_cnn = F.dropout(F.relu(title_cnn), p=self.config.dropout).transpose(3, 2)

        # bs * browsed_num * filter_num
        title_att = self.title_attention(title_cnn)

        # abstract
        # bs * browsed_num * abstract_len * embedding_dim
        abstract_embedded = F.dropout(self.word_embedding(abstract), p=self.config.dropout)

        # bs * browsed_num * filter_num * abstract_len
        abstract_cnn = self.title_abstract_cnn(abstract_embedded) #########
        
        # bs * abstract_len * filter_num
        abstract_cnn = F.dropout(F.relu(abstract_cnn), p=self.config.dropout).transpose(1, 2)
        # bs * filter_num
        abstract_att = self.abstract_attention(abstract_cnn)

        # topic
        # bs * topic_embedding_dim
        topic_embedded = self.topic_embedding(topic).squeeze(dim=1)
        # bs * filter_num
        topic_dense = F.relu(self.topic_dense(topic_embedded))

        # subtopic
        # bs * topic_embedding_dim
        subtopic_embedded = self.subtopic_embedding(subtopic).squeeze(dim=1)
        # bs * filter_num
        subtopic_dense = F.relu(self.subtopic_dense(subtopic_embedded))

        # news_r
        # bs * 4 * filter_num
        att_input = torch.stack([title_att, abstract_att, topic_dense, subtopic_dense], dim=1)

        # bs * filter_num
        news_r = self.view_attention(att_input)
        return news_r

class NAML_UserEncoder(nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.config = config
        self.attention = AdditiveAttention(config.query_dim, config.filter_num)
    
    def forward(self, news):
        # bs * browsed_num * filter_num
        user_r = self.attention(news)
        return user_r

class NAML(nn.Module):
    def __init__(self, config, pretrained_embedding=None):
        super(NAML, self).__init__()
        self.config = config
        self.pretrained_embedding = pretrained_embedding
        self.NewsEncoder = NewsEncoder(config, pretrained_embedding)
        self.UserEncoder = UserEncoder(config)
    
    def forward(self, data):

        #bs * 1
        topic = data[0]
        subtopic = data[1]
        #bs * browsed_num * title_words_num
        history = data[2]
        #bs * abstract_len
        abstract = data[3]
        #bs * 1
        candi_topic = data[4]
        candi_subtopic = data[5]
        #bs * candidate_num * title_words_num
        candidate = date[6]
        candidate_abstract = data[7]

        # bs * candidate_num *filter_num
        candidate_news_r = self.NewsEncoder(candi_topic, candi_subtopic, candidate, candidate_abstract)
        # bs * browsed_num * filter_num
        browsed_news_r = self.NewsEncoder(topic, subtopic, history, abstract)
        # bs * filternum
        user_r = self.UserEncoder(browsed_news_r)
        # batch_size, 1+K
        click_prob = torch.stack([torch.bmm(user_r.unsqueeze(dim=1), x.unsqueeze(dim=2)).flatten()
                    for x in candidate_news_r], dim=1)
        return click_prob
    
    def get_news_r(self, title):
        return self.NewsEncoder(title)
    
    def get_user_r(self, browsed_news_r):
        return self.UserEncoder(browsed_news_r.to(device))
    
    def test(self, user_r, candidate_news_r):
        return torch.bmm(user_r.to(device).unsqueeze(dim=1), candidate_news_r.to(device).unsqueeze(dim=2)).flatten()