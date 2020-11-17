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
        self.batch_num = self.datalen // self.batch_size + 1
    
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

        return {'userId':torch.Tensor(id_batch).long(), 
                'topic': torch.Tensor(topic_batch).long(),
                'sub-topic': torch.Tensor(subtopic_batch).long(),
                'history':torch.Tensor(history_batch).long(), 
                'candidate': torch.Tensor(candidate_batch).long(),
                'candi-topic': torch.Tensor(canditopic_batch).long(),
                'candi-subtopic': torch.Tensor(candisubtopic_batch).long(),
                'abstract': torch.Tensor(abstract_batch).long(),
                'candi-abstract': torch.Tensor(candi_abstract_batch).long()}


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

        return {'userId':torch.Tensor(id_batch).long(), 
                'topic': torch.Tensor(topic_batch).long(),
                'sub-topic': torch.Tensor(subtopic_batch).long(),
                'history':torch.Tensor(history_batch).long(), 
                'candidate': torch.Tensor(candidate_batch).long(),
                'candi-topic': torch.Tensor(canditopic_batch).long(),
                'candi-subtopic': torch.Tensor(candisubtopic_batch).long(),
                'label': torch.Tensor(label_batch).long()}
                # 'abstract': torch.Tensor(abstract_batch).long(),
                # 'candi-abstract': torch.Tensor(candi_abstract_batch).long()}

    def getImpressionIndex(self):
        return self.impression_index
    
    def getLabels(self):
        return self.label
