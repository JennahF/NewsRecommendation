import torch

class myTrainDataLoader():
    def __init__(self, data, batch_size):
        super(myTrainDataLoader, self).__init__()

        self.uid = data[0]
        self.topic = data[1]
        self.subtopic = data[2]
        self.history = data[3]
        self.abstract = data[4]
        self.candi_topic = data[5]
        self.candi_subtopic = data[6]
        self.candidate = data[7]
        self.candi_abstract = data[8]

        self.datalen = len(self.uid)
        self.batch_size = batch_size
        
        if self.datalen % self.batch_size == 0:
            self.batch_num = (self.datalen // self.batch_size)
        else:
            self.batch_num = (self.datalen // self.batch_size) + 1
    
    def getSandE(self, index):
        start = index * self.batch_size
        if start + self.batch_size >= self.datalen:
            end = self.datalen
        else:
            end = start + self.batch_size
        return start, end

    def __getitem__(self, index):
        assert index < self.batch_num
        start, end = self.getSandE(index)
        id_batch = self.uid[start:end]
        history_batch = self.history[start:end]
        candidate_batch = self.candidate[start:end]
        topic_batch = self.topic[start:end]
        subtopic_batch = self.subtopic[start:end]
        canditopic_batch = self.candi_topic[start:end]
        candisubtopic_batch = self.candi_subtopic[start:end]
        abstract_batch = self.abstract[start:end]
        candi_abstract_batch = self.candi_abstract[start:end]

        print(len(history_batch), start, end)

        return {'userId':torch.LongTensor(id_batch), 
                'topic': torch.LongTensor(topic_batch),
                'sub-topic': torch.LongTensor(subtopic_batch),
                'history':torch.LongTensor(history_batch), 
                'candidate': torch.LongTensor(candidate_batch),
                'candi-topic': torch.LongTensor(canditopic_batch),
                'candi-subtopic': torch.LongTensor(candisubtopic_batch),
                'abstract': torch.LongTensor(abstract_batch),
                'candi-abstract': torch.LongTensor(candi_abstract_batch)}


class myTestDataLoader():
    def __init__(self, data, batch_size):
        super(myTestDataLoader, self).__init__()
        
        self.uid = data[0]
        self.topic = data[1]
        self.subtopic = data[2]
        self.history = data[3]
        self.abstract = data[4]
        self.candi_topic = data[5]
        self.candi_subtopic = data[6]
        self.candidate = data[7]
        self.candi_abstract = data[8]
        self.label = data[9]
        self.impression_index = data[10]
        
        self.datalen = len(self.uid)
        self.batch_size = batch_size
        if self.datalen % self.batch_size == 0:
            self.batch_num = (self.datalen // self.batch_size)
        else:
            self.batch_num = (self.datalen // self.batch_size) + 1

        print('batch_size: ', self.batch_size)
    
    def getSandE(self, index):
        assert index < self.batch_num
        start = index * self.batch_size
        if start + self.batch_size >= self.datalen:
            end = self.datalen
        else:
            end = start + self.batch_size
        return start, end

    def __getitem__(self, index):
        start, end = self.getSandE(index)
        # print(index, start, end)
        id_batch = self.uid[start:end]
        history_batch = self.history[start:end]
        candidate_batch = self.candidate[start:end]
        label_batch = self.label[start:end]
        # print(candidate_batch)
        # print(len(candidate_batch[0]))
        topic_batch = self.topic[start:end]
        subtopic_batch = self.subtopic[start:end]
        canditopic_batch = self.candi_topic[start:end]
        candisubtopic_batch = self.candi_subtopic[start:end]
        # abstract_batch = self.abstract[start:end]
        # candi_abstract_batch = self.candi_abstract[start:end]

        return {'userId':torch.LongTensor(id_batch), 
                'topic': torch.LongTensor(topic_batch),
                'sub-topic': torch.LongTensor(subtopic_batch),
                'history':torch.LongTensor(history_batch), 
                'candidate': torch.LongTensor(candidate_batch),
                'candi-topic': torch.LongTensor(canditopic_batch),
                'candi-subtopic': torch.LongTensor(candisubtopic_batch),
                'label': torch.LongTensor(label_batch)}
                # 'abstract': torch.LongTensor(abstract_batch),
                # 'candi-abstract': torch.LongTensor(candi_abstract_batch)}

    def getImpressionIndex(self):
        return self.impression_index
    
    def getLabels(self):
        return self.label

class myNewsLoader():
    def __init__(self, news_vec, model_name, batch_size):
        super(myNewsLoader, self).__init__()
        '''
        {
            'Nxxxx': {'topic': xxx, 'subtopic': xxx, 'title': [xxx,xxx,...], 'abstract': [xxx,xxx,...]},
            ...
        }
        '''
        self.news_ids = list(news_vec.keys())
        self.news_infos = list(news_vec.values())
        self.model_name = model_name
        self.batch_size = batch_size
        self.datalen = len(news_vec)
        if len(news_vec) % self.batch_size == 0:
            self.batch_num = len(news_vec) // self.batch_size
        else:
            self.batch_num = len(news_vec) // self.batch_size + 1

    def getSandE(self, index):
        assert index < self.batch_num
        start = index * self.batch_size
        if start + self.batch_size >= self.datalen:
            end = self.datalen
        else:
            end = start + self.batch_size
        return start, end

    def __getitem__(self, index):
        start, end = self.getSandE(index)
        ids = self.news_ids[start:end]
        infos = self.news_infos[start:end]
        topics = []
        subtopics = []
        titles = []
        abstracts = []
        #[bs * title_words_num]
        titles = [x['title'] for x in infos]

        if self.model_name == 'NRMS':
            return ids, titles

        if self.model_name == 'LSTUR':
            topics = [[x['topic']] for x in infos]
            subtopics = [[x['subtopic']] for x in infos]
            return ids, [topics, subtopics, titles]

        if self.model_name == 'NAML':
            topics = [[x['topic']] for x in infos]
            subtopics = [[x['subtopic']] for x in infos]
            abstracts = [x['abstract'] for x in infos]
            return ids, [topics, subtopics, titles, abstracts]

class myUserLoader():
    def __init__(self, encoded_news, userid, history, model_name, batch_size):
        super(myUserLoader, self).__init__()
        '''
        encoded_news:
        {
            'Nxxxx': [xxx](embedding_size),
            ...
        }
        userid:
        [xxx, xxx, ...]
        history:
        [
            ['Nxxxx', 'Nxxxx', ...],
            ...
        ]
        '''
        self.encoded_news = encoded_news
        self.userid = userid
        self.history = history
        self.model_name = model_name
        self.batch_size = batch_size
        self.datalen = len(history)
        if len(history) % self.batch_size == 0:
            self.batch_num = len(history) // self.batch_size
        else:
            self.batch_num = len(history) // self.batch_size + 1


    def getSandE(self, index):
        assert index < self.batch_num
        start = index * self.batch_size
        if start + self.batch_size >= self.datalen:
            end = self.datalen
        else:
            end = start + self.batch_size
        return start, end
    
    def __getitem__(self, index):
        start, end = self.getSandE(index)
        history = []
        for i in range(start, end):
            his = [self.encoded_news[newsid] for newsid in self.history[i]]
            history.append(his)
        if self.model_name != 'LSTUR':
            return history
        else:
            userid = self.userid[start:end]
            userid = [[x] for x in userid]
            return history, userid

class myCandidateLoader():
    def __init__(self, candidate_vec, encoded_user, encoded_news, batch_size):
        super(myCandidateLoader, self).__init__()
        '''
        candidate_vec:
        [
            ['Nxxxx', 'Nxxxx'],
            ...
        ],
        encoded_user:
        [
            [xxx],(embedding_size)
            [xxx]
        ],
        encoded_news:
        [
            'Nxxxx':[xxx](embedding_size),
            ...
        ]
        '''    
        self.candidate_vec = candidate_vec
        self.encoded_user = encoded_user
        self.encoded_news = encoded_news
        self.batch_size = batch_size
        self.datalen = len(candidate_vec)
        self.batch_num = len(candidate_vec) // batch_size if len(candidate_vec % batch_size == 0) else len(candidate_vec) // batch_size + 1

    def getSandE(self, index):
        assert index < self.batch_num
        start = index * self.batch_size
        if start + self.batch_size >= self.datalen:
            end = self.datalen
        else:
            end = start + self.batch_size
        return start, end

    def __getitem__(self, data):
        start, end = self.getSandE(index)
        candidate = []
        user = self.encoded_user[start:end]
        for i in range(start, end):
            candi = [self.encoded_news[candid] for candid in self.candidate_vec[i]]
            candidate.append(candi)
        return candidate, user


