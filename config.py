class NRMSconfig():
    neg_sample_num = 1
    word_embedding_dim = 300
    title_max_words_num = 48
    browsed_max_num = 50 #
    selfatt_head = 15
    head_output_dim = 16
    selfatt_output_dim = 20
    query_dim = 200
    vocab_size = 53304 + 1
    dropout = 0.2
    candidate_max_num = 50 #

class LSTURconfig():
    vocab_size = 53304 + 1
    word_embedding_dim = 200
    # topic_embedding_dim = 
    topic_num = 18 + 1
    sub_topic_num = 268+1
    topic_embedding_dim = 100
    subtopic_embedding_dim = 100
    windows_size = 3
    filter_num = 300
    query_dim = 200
    dropout = 0.2
    mask_prob = 0.5
    user_embedding_dim = 94057 + 1
    browsed_max_num = 50 #

class NAMLconfig():
    query_dim = 200
    filter_num = 400
    vocab_size = 98177 + 1
    embedding_dim = 300
    topic_num = 18 + 1
    sub_topic_num = 268+1
    topic_embedding_dim = 100
    window_size = 3
    browsed_max_num = 50 #
    dropout = 0.2
    abstract_max_len = 200
