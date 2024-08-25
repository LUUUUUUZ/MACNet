# -*- coding: utf-8 -*-
import  math
import os
import pickle
from tqdm import tqdm
import copy

import torch
import numpy as np
from collections import Counter

class OnlineItemSimilarity:

    def __init__(self, device, num_items,**kwargs):
        self.item_size = num_items
        self.item_embeddings = None
        self.device = device
        self.total_item_list = torch.tensor([i for i in range(self.item_size)],
                                            dtype=torch.long)
                
    def update_embedding_matrix(self, item_embeddings):
        
        print("update embedding matrix!!!!")
        
        self.item_embeddings = copy.deepcopy(item_embeddings).cpu()
        self.base_embedding_matrix =self.item_embeddings(self.total_item_list)
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()
        # cached epoch level
        print('cached embedding similarity on Epoch level')
        self.similarity_dict = self._generate_item_similarity() 

    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -1, 100
        for item_idx in range(1, self.item_size):
            try:
                item_vector = self.item_embeddings(torch.tensor(item_idx)).view(-1, 1)
                item_similarity = torch.mm(self.base_embedding_matrix, item_vector).view(-1)
                max_score = max(torch.max(item_similarity), max_score)
                min_score = min(torch.min(item_similarity), min_score)
            except:
                print("ssssss")
                continue
        return max_score, min_score
        
    def _generate_item_similarity(self):
        similar_dict = {}
        for item_idx in tqdm(range(self.item_size)):
            similar_dict[item_idx] = self.most_similar_one(item_idx)
        return similar_dict
    
    def most_similar(self, item_idx):
        return self.similarity_dict[item_idx]

    def most_similar_one(self, item_idx):
        
        item_idx = torch.tensor(item_idx, dtype=torch.long)
        item_vector = self.item_embeddings(item_idx).view(-1, 1)
        item_similarity = torch.mm(self.base_embedding_matrix, item_vector).view(-1)
        item_similarity = (item_similarity - self.min_score) / (self.max_score - self.min_score)
        #remove item idx itself
        values, indices = item_similarity.topk(2)

        item_list = indices.tolist()
        if item_idx in item_list:
            item_list.remove(item_idx)
        return item_list[0]

class OfflineItemSimilarity:
    def __init__(self, df, similarity_path, similarity_method='ItemCF', **kwargs):
        
        self.similarity_path = similarity_path
        # train_data_list used for ItemTrans, train_data_dict used for itemCF and itemCF-IUF
        self.train_data_list, self.train_item_list, self.train_data_dict = self._load_train_data(df)
        self.method = similarity_method
        self.similarity_dict = self.load_similarity_model(self.similarity_path)
    
    def _convert_data_to_dict(self, data):
        """
        split the data set
        testdata is a test data set
        traindata is a train set
        """
        train_data_dict = {}
        for user,item,record in data:
            train_data_dict.setdefault(user,{})
            train_data_dict[user][item] = record
        return train_data_dict

    def _save_dict(self, dict_data, save_path = './similarity.pkl'):
        print("saving data to ", save_path)
        with open(save_path, 'wb') as write_file:
            pickle.dump(dict_data, write_file)

    def _load_train_data(self, df):
        """
        read the data from the data file which is a data set
        """
        # import pdb;pdb.set_trace()
        train_data = []
        train_data_list = []
        train_data_set_list = []
        for index, row in df.iterrows():
            userid= row['userId']
            items = row['items']
            train_data_list.append(items)
            train_data_set_list += items
            for itemid in items:
                train_data.append((userid,itemid,int(1)))
        return train_data_list, set(train_data_set_list), self._convert_data_to_dict(train_data) 

    def _generate_item_similarity(self, train=None, save_path='./'):
        """
        calculate co-rated users between items
        """
        
        print("getting item similarity...")
        train = train or self.train_data_dict
        C = dict()
        N = dict()

        if self.method == 'ItemTrans':
            print("Step 1: Compute Statistics")
            transits = Counter()
            for seq in self.train_data_list:
                transits.update(zip(seq,seq[1:]))
            
            prev_i, next_i = zip(*transits.keys())
            transits_cnts = np.array(list(transits.values()))
            for idx in range(len(prev_i)):
                i = prev_i[idx]
                C.setdefault(i,{})
                
                j = next_i[idx]
                C[i].setdefault(j,0)
                C[i][j] += transits_cnts[idx]       

        elif self.method in ['ItemCF', 'ItemCF_IUF']:
            print("Step 1: Compute Statistics")
            data_iter = tqdm(enumerate(train.items()), total=len(train.items()))
            for idx, (u, items) in data_iter:
                if self.method == 'ItemCF':
                    for i in items.keys():
                        N.setdefault(i,0)
                        N[i] += 1
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i,{})
                            C[i].setdefault(j,0)
                            C[i][j] += 1
                elif self.method == 'ItemCF_IUF':
                    for i in items.keys():
                        N.setdefault(i,0)
                        N[i] += 1
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i,{})
                            C[i].setdefault(j,0)
                            C[i][j] += 1 / math.log(1 + len(items) * 1.0)
        
        print("Step 2: Compute co-rate matrix")
        self.itemSimBest = dict()
        c_iter = tqdm(enumerate(C.items()), total=len(C.items()))
        for idx, (cur_item, related_items) in c_iter:
            max_score = 0
            max_score_related_item = -1
            for related_item, score in related_items.items():
                if self.method == 'ItemTrans':
                    this_score = score
                else:
                    this_score = score / math.sqrt(N[cur_item] * N[related_item])
                if this_score > max_score:
                    max_score = this_score
                    max_score_related_item = related_item
            self.itemSimBest[cur_item] = max_score_related_item 
            # only saved most similiar
        self._save_dict(self.itemSimBest, save_path=save_path)


    def load_similarity_model(self, similarity_model_path):
        if not similarity_model_path:
            raise ValueError('invalid path')
        elif not os.path.exists(similarity_model_path):
            print("the similirity dict not exist, generating...")
            self._generate_item_similarity(save_path=self.similarity_path)
        if self.method in ['ItemTrans', 'ItemCF', 'ItemCF_IUF']:
            with open(similarity_model_path, 'rb') as read_file:
                similarity_dict = pickle.load(read_file)
            return similarity_dict

    def most_similar(self, item):
        if item not in self.similarity_dict.keys():
            return item
        else:
            return self.similarity_dict[item]

