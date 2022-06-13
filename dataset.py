from torch.utils.data import Dataset
from torch import nn
import torch
import numpy as np
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
import os
import torch
import pandas as pd

class recipe_dataset(Dataset):
    def __init__(self, train, test, recipe_ingredient, u_dim):

        self.train = train
        self.test = test
        self.r_list = train
        self.recipe_ingredient = recipe_ingredient    
    
        self.users = train.유저_아이디.unique()
        self.recipes = list(set(train.레시피_아이디.unique().tolist() + test.레시피_아이디.unique().tolist()))
        self.ingredients = np.load('/HGAT/ingredient_emb/ing_seq.npy')
        print('사용할 유저 수 :',len(self.users))
        print('사용할 레시피 수 :',len(self.recipes)) 
        print('사용할 재료 수:', len(self.ingredients))

        self.user2idx = { v:k for k,v in enumerate(self.users)}
        self.idx2user = {k:v for k,v in enumerate(self.users)}
        
        self.recipe2idx = {v:k for k,v in enumerate(self.recipes)}
        self.idx2recipe = {k:v for k,v in enumerate(self.recipes)}
        
        self.ing2idx = {v:k for k,v in enumerate(self.ingredients)}
        self.idx2ing = {k:v for k,v in enumerate(self.ingredients)}       

        self.user_embedding = nn.Embedding(len(self.users), u_dim)
        self.recipe_embedding = nn.Embedding.from_pretrained(torch.load('/HGAT/sentence_emb/sentence_5900_emb_multilingual-cased-v1.pt').squeeze(1))
        self.ing_embedding = nn.Embedding.from_pretrained(torch.load('/HGAT/ingredient_emb/ing_emb.pt').squeeze(1))

    def get_recipe2idx(self):
        return self.recipe2idx

    def get_idx2recipe(self):
        return self.idx2recipe
    
    def get_user2idx(self):
        return self.user2idx

    def get_idx2user(self):
        return self.idx2recipe
    
    def get_recipecat2id(self):
        return self.recipecat2id

    def get_user_recipe_ing(self):
        return self.users, self.recipes, self.ingredients

    def get_matrix(self):        
        # 유저_레시피

        u_r = torch.zeros(len(self.users) ,len(self.recipes))
        for u in self.users:
            u_idx = self.user2idx[u]                                        
            for r in self.train[self.train.유저_아이디==u].레시피_아이디.values:
                r_idx = self.recipe2idx[r]
                u_r[u_idx][r_idx] = 1

        user_rel_matrix = [u_r]

        ######### 레시피 #############
        recipe_rel_matrix = []
        
        r_u = torch.zeros(len(self.recipes), len(self.users))
        r_i = torch.zeros(len(self.recipes), len(self.ingredients))
        
        for r in self.recipes:
            r_idx = self.recipe2idx[r]
            
            # 레시피_유저
            for u in self.train[self.train.레시피_아이디==r].유저_아이디.values:
                u_idx = self.user2idx[u]                                        
                r_u[r_idx][u_idx] = 1
            
            # 레시피_재료
            for i in self.recipe_ingredient[self.recipe_ingredient.레시피_아이디==r].재료_아이디.values:
                i_idx = self.ing2idx[i]
                r_i[r_idx][i_idx] = 1
        
        # 레시피_레시피        
        r_r_cos = torch.load('/HGAT/sentence_emb/sentence_5900_cos_multilingual-cased-v1.pt')

        r_r_cos[r_r_cos>0.84] = 1
        r_r_cos[r_r_cos<=0.84] = 0            

        recipe_rel_matrix.append(r_r_cos)        
        recipe_rel_matrix.append(r_u)    
        recipe_rel_matrix.append(r_i)          
        

        return user_rel_matrix, recipe_rel_matrix
    