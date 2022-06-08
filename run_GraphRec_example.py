import copy
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os
from os import path
from utils import utils

"""
GraphRec: Graph Neural Networks for Social Recommendation. 
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 
In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. Preprint[https://arxiv.org/abs/1902.07243]

If you use this code, please cite our paper:
```
@inproceedings{fan2019graph,
  title={Graph Neural Networks for Social Recommendation},
  author={Fan, Wenqi and Ma, Yao and Li, Qing and He, Yuan and Zhao, Eric and Tang, Jiliang and Yin, Dawei},
  booktitle={WWW},
  year={2019}
}
```

"""


class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v_history(nodes_v)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)


def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0
    return 0


def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae

def get_top_k_recommendations(model, device, dataset_name, target_users, history_u_lists, history_v_lists, k):
    B, users, items = utils.create_user_item_bipartite_graph(history_u_lists)

    user_communities_interactions_dict_filepath = './results/' + dataset_name + '/user_communities_interactions_dict.pickle'
    item_community_dict_filepath = './results/' + dataset_name + '/item_community_dict.pickle'

    if path.exists() and path.exists('./results/' + dataset_name + '/user_communities_interactions_dict.pickle'):
        user_communities_interactions_dict = pickle.load(user_communities_interactions_dict_filepath)
        item_community_dict = pickle.load(item_community_dict_filepath)

        with open(user_communities_interactions_dict_filepath, 'wb') as handle:
            pickle.dump(user_communities_interactions_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(item_community_dict_filepath, 'wb') as handle:
            pickle.dump(item_community_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        user_communities_interactions_dict, item_community_dict = utils.create_user_communities_interaction_dict(B, items, history_u_lists)

    model.eval()
    all_items = list(set(history_v_lists.keys()))
    # {user_id:[item_id1, ..., item_idk]}
    results = {}
    with torch.no_grad():
        for user_id in target_users:
            if user_id not in history_u_lists:
                continue
            candidate_items = [item_id for item_id in all_items if item_id not in history_u_lists[user_id]]
            test_u = torch.tensor(np.repeat(user_id, len(candidate_items))).to(device)
            test_v = torch.tensor(candidate_items).to(device)
            # multiply this with the mask of excluded recommendations derived from target_users_items
            val_output = model.forward(test_u, test_v).data.cpu().numpy()
            print(len(val_output))
            topk_prediction_indices = np.argpartition(val_output, -k)[-k:]
            topk_prediction_indices_sorted = list(np.flip(topk_prediction_indices[np.argsort(val_output[topk_prediction_indices])]))
            topk_item_ids = [candidate_items[i] for i in topk_prediction_indices_sorted]

            user_item_communities = [item_community_dict[item_id] for item_id in history_u_lists[user_id]]
            user_diversity = utils.entropy_label_distribution(user_item_communities)

            recommended_item_communities = [item_community_dict[item_id] for item_id in topk_item_ids]
            entropy_item_diversity = utils.entropy_label_distribution(recommended_item_communities)
            weighted_average_item_diversity = utils.calculate_weighted_average_diversity(user_communities_interactions_dict[user_id])

            results[user_id] = {
                'recommendations' : topk_item_ids,
                'user_diversity' : user_diversity,
                'entropy_item_diversity' : entropy_item_diversity,
                'weighted_average_item_diversity' : weighted_average_item_diversity,
            }

            print(results[user_id])

    return results

def train_and_store_model(model, epochs, device, train_loader, test_loader, lr, dataset_name):
    """
    ## toy dataset 
    history_u_lists, history_ur_lists:  user's purchased history (item set in training set), and his/her rating score (dict)
    history_v_lists, history_vr_lists:  user set (in training set) who have interacted with the item, and rating score (dict)

    train_u, train_v, train_r: training_set (user, item, rating)
    test_u, test_v, test_r: testing set (user, item, rating)

    # please add the validation set

    social_adj_lists: user's connected neighborhoods
    ratings_list: rating value from 0.5 to 4.0 (8 opinion embeddings)
    """
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0

    for epoch in range(1, epochs + 1):

       train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae)
       expected_rmse, mae = test(model, device, test_loader)
       # please add the validation set to tune the hyper-parameters based on your datasets.

       if not os.path.exists('./checkpoint/' + dataset_name):
           os.makedirs('./checkpoint/' + dataset_name)
    # early stopping (no validation set in toy dataset)
       if best_rmse > expected_rmse:
           best_rmse = expected_rmse
           best_mae = mae
           endure_count = 0
           best_model = copy.deepcopy(model)
           torch.save(best_model.state_dict(), './checkpoint/' + dataset_name + '/model')
       else:
           endure_count += 1
       print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))

       rmse_mae_dict = {
           'rmse' : expected_rmse,
           'mae' : mae
       }

       with open('./results/' + dataset_name + '/rmse_mae.pickle', 'wb') as handle:
           pickle.dump(rmse_mae_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

       if endure_count > 5:
           break


def evaluate_and_store_recommendations(model, device, dataset_name, test_u, history_u_lists, history_v_lists, k):
    target_users = list(set(test_u))
    results = get_top_k_recommendations(model, device, dataset_name, target_users, history_u_lists, history_v_lists, k)

    return results


def main():

    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--k', type=int, default=10, metavar='N', help='number of recommendations to generate per user')
    parser.add_argument('--gpu_id', type=int, default=0, metavar='N', help='gpu id')
    parser.add_argument('--dataset_name', type=str, default='toy_dataset', help='dataset name')
    parser.add_argument('--load_model', type=bool, default=True, help='if this is False, then the model is trained from scratch')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    dir_data = './data/' + args.dataset_name + '/'

    path_data = dir_data + args.dataset_name + ".pickle"
    data_file = open(path_data, 'rb')

    train_filepath = './data/' + args.dataset_name + '/train.tsv'
    test_filepath = './data/' + args.dataset_name + '/test.tsv'
    social_connections_filepath = './data/' + args.dataset_name + '/social_connections.tsv'
    train_dict = utils.create_user_item_rating_dict_from_file(train_filepath)
    test_dict = utils.create_user_item_rating_dict_from_file(test_filepath)

    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, \
    test_u, test_v, test_r, social_adj_lists, ratings_list = utils.preprocess_data_test(train_dict, test_dict, social_connections_filepath)

    
    # history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, \
    # test_u, test_v, test_r, social_adj_lists, ratings_list = pickle.load(data_file)

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()

    u2e = nn.Embedding(num_users, args.embed_dim).to(device)
    v2e = nn.Embedding(num_items, args.embed_dim).to(device)
    r2e = nn.Embedding(num_ratings, args.embed_dim).to(device)

    # user feature
    # features: item * rating
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, args.embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, args.embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device,
                               uv=True)
    # neighbors
    agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e, args.embed_dim, cuda=device)
    enc_u = Social_Encoder(lambda nodes: enc_u_history(nodes).t(), args.embed_dim, social_adj_lists, agg_u_social,
                           base_model=enc_u_history, cuda=device)

    # item feature: user * rating
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, args.embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, args.embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device,
                               uv=False)

    # model
    model = GraphRec(enc_u, enc_v_history, r2e).to(device)

    if args.load_model is False:
        train_and_store_model(model, args.epochs, device, train_loader, test_loader, args.lr, args.dataset_name)

    model.load_state_dict(torch.load('./checkpoint/' + args.dataset_name + '/model'))
    model.eval()

    results = evaluate_and_store_recommendations(model, device, args.dataset_name, test_u, history_u_lists, history_v_lists, args.k)

    with open('./results/' + args.dataset_name + '/recommendations.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    unique_recommended_items = set()
    for user_id in results:
        unique_recommended_items.update(results[user_id]['recommendations'])

    users_items_stats = {
        'num_users' : num_users,
        'num_items' : num_items,
        'num_recommended_items' : unique_recommended_items,
        'item_coverage' : round(unique_recommended_items/num_items, 2)
    }

    with open('./results/' + args.dataset_name + '/users_items_stats.pickle', 'wb', 'wb') as handle:
        pickle.dump(users_items_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
