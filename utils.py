import time
import pandas as pd
import torch
from torch.utils.data import Dataset


def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())


def load_embedding(word2vec_file):
    with open(word2vec_file, encoding='utf-8') as f:
        word_emb = list()
        word_dict = dict()
        word_emb.append([0])
        word_dict['<UNK>'] = 0
        for line in f.readlines():
            tokens = line.split(' ')
            word_emb.append([float(i) for i in tokens[1:]])
            word_dict[tokens[0]] = len(word_dict)
        word_emb[0] = [0] * len(word_emb[1])
    return word_emb, word_dict

def predict_mse(model, dataloader, device):
    mse, sample_count = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            user_reviews, item_reviews, ratings = map(lambda x: x.to(device), batch)
            predict = model(user_reviews, item_reviews)
            mse += torch.nn.functional.mse_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
    return mse / sample_count  # dataloader上的均方误差


class DeepCoNNDataset(Dataset):
    def __init__(self, data_path, word_dict, config, retain_rui=True):
        self.word_dict = word_dict
        self.config = config
        self.retain_rui = retain_rui  # 是否在最终样本中，保留user和item的公共review
        self.PAD_WORD_idx = self.word_dict[config.PAD_WORD]
        self.review_length = config.review_length
        self.review_count = config.review_count
        self.lowest_r_count = config.lowest_review_count  # lowest amount of reviews wrote by exactly one user/item

        df = pd.read_csv(data_path, header=None, names=['userID', 'itemID', 'review', 'rating'])
        df['review'] = df['review'].apply(self._review2id)  # 分词->数字
        self.sparse_idx = set()  # 暂存稀疏样本的下标，最后删除他们
        user_reviews = self._get_reviews(df)  # 收集每个user的评论列表
        item_reviews = self._get_reviews(df, 'itemID', 'userID')
        rating = torch.Tensor(df['rating'].to_list()).view(-1, 1)

        self.user_reviews = user_reviews[[idx for idx in range(user_reviews.shape[0]) if idx not in self.sparse_idx]]
        self.item_reviews = item_reviews[[idx for idx in range(item_reviews.shape[0]) if idx not in self.sparse_idx]]
        self.rating = rating[[idx for idx in range(rating.shape[0]) if idx not in self.sparse_idx]]

    def __getitem__(self, idx):
        return self.user_reviews[idx], self.item_reviews[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]

    def _get_reviews(self, df, lead='userID', costar='itemID'):
        # 对于每条训练数据，生成用户的所有评论汇总
        reviews_by_lead = dict(list(df[[costar, 'review']].groupby(df[lead])))  # 每个user/item评论汇总
        lead_reviews = []
        for idx, (lead_id, costar_id) in enumerate(zip(df[lead], df[costar])):
            df_data = reviews_by_lead[lead_id]  # 取出lead的所有评论：DataFrame
            if self.retain_rui:
                reviews = df_data['review'].to_list()  # 取lead所有评论：列表
            else:
                reviews = df_data['review'][df_data[costar] != costar_id].to_list()  # 不含lead与costar的公共评论
            if len(reviews) < self.lowest_r_count:
                self.sparse_idx.add(idx)
            reviews = self._adjust_review_list(reviews, self.review_length, self.review_count)
            lead_reviews.append(reviews)
        return torch.LongTensor(lead_reviews)

    def _adjust_review_list(self, reviews, r_length, r_count):
        reviews = reviews[:r_count] + [[self.PAD_WORD_idx] * r_length] * (r_count - len(reviews))  # 评论数量固定
        reviews = [r[:r_length] + [0] * (r_length - len(r)) for r in reviews]  # 每条评论定长
        return reviews

    def _review2id(self, review):  # 将一个评论字符串分词并转为数字
        if not isinstance(review, str):
            return []  # 貌似pandas的一个bug，读取出来的评论如果是空字符串，review类型会变成float
        wids = []
        for word in review.split():
            if word in self.word_dict:
                wids.append(self.word_dict[word])  # 单词映射为数字
            else:
                wids.append(self.PAD_WORD_idx)
        return wids
