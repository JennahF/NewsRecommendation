import re

def remove_sig(str:str):
    '''remove_sig, remove signals from the input string
    Args:
        str: the input string

    Returns:
        A string without signals like .'", etc
    '''
    return re.sub("[+\.\!\/<>“”''"
                  ",$?\-%^*():+\"\']+|[+——！，。？、~#￥%……&*（）]+", "", str.strip())

class Dictionary:
    def __init__(self, word2id):
        self.word2id = word2id
        self.id2word = {id: word for word, id in word2id.items()}
        self.vocab_len = len(self.word2id)
    
    def getId(self, word):
        return self.word2id[remove_sig(word)]
