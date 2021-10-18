import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


class CNN(nn.Module):

    def __init__(self, config, word_dim):
        super(CNN, self).__init__()

        self.kernel_count = config.kernel_count
        self.review_count = config.review_count

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=word_dim,
                out_channels=config.kernel_count,
                kernel_size=config.kernel_size,
                padding=(config.kernel_size - 1) // 2),  # out shape(new_batch_size, kernel_count, review_length)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, config.review_length)),  # out shape(new_batch_size,kernel_count,1)
            nn.Dropout(p=config.dropout_prob))

        self.linear = nn.Sequential(
            nn.Linear(config.kernel_count * config.review_count, config.cnn_out_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_prob))

    def forward(self, vec):  # input shape(new_batch_size, review_length, word2vec_dim)
        latent = self.conv(vec.permute(0, 2, 1))  # out(new_batch_size, kernel_count, 1) kernel count指一条评论潜在向量
        latent = self.linear(latent.reshape(-1, self.kernel_count * self.review_count))
        return latent  # out shape(batch_size, cnn_out_dim)


class FactorizationMachine(nn.Module):

    def __init__(self, p, k):  # p=cnn_out_dim
        super(FactorizationMachine, self).__init__()
        self.v = nn.Parameter(torch.zeros(p, k))
        self.linear = nn.Linear(p, 1, bias=True)

    def forward(self, x):
        linear_part = self.linear(x)  # input shape(batch_size, cnn_out_dim), out shape(batch_size, 1)
        inter_part1 = torch.mm(x, self.v)
        inter_part2 = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(inter_part1 ** 2 - inter_part2, dim=1)
        output = linear_part.transpose(1, 0) + 0.5 * pair_interactions
        return output.view(-1, 1)  # out shape(batch_size, 1)


class DeepCoNN(nn.Module):

    def __init__(self, config, word_emb):
        super(DeepCoNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        self.cnn_u = CNN(config, word_dim=self.embedding.embedding_dim)
        self.cnn_i = CNN(config, word_dim=self.embedding.embedding_dim)
        self.fm = FactorizationMachine(config.cnn_out_dim * 2, 10)

    def forward(self, user_review, item_review):  # input shape(batch_size, review_count, review_length)
        new_batch_size = user_review.shape[0] * user_review.shape[1]
        user_review = user_review.reshape(new_batch_size, -1)
        item_review = item_review.reshape(new_batch_size, -1)

        u_vec = self.embedding(user_review)
        i_vec = self.embedding(item_review)

        user_latent = self.cnn_u(u_vec)
        item_latent = self.cnn_i(i_vec)

        concat_latent = torch.cat((user_latent, item_latent), dim=1)
        prediction = self.fm(concat_latent)
        return prediction
