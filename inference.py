import pandas as pd
import os
from models import HGAT
import torch
from torch import nn
from tqdm import tqdm
from dataset import recipe_dataset
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch.utils.data import BatchSampler, RandomSampler

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--user_dimension', type=int, default=16, help='Random seed.')
parser.add_argument('--recipe_dimension', type=int, default=512, help='Random seed.')
parser.add_argument('--ingredient_dimension', type=int, default=512, help='Random seed.')
parser.add_argument('--hidden_unit', type=int, default=4, help='Random seed.')
parser.add_argument('--n_heads', type=int, default=4, help='Random seed.')

parser.add_argument('--alpha', type=float, default=0.1, help='Random seed.')
parser.add_argument('--drop_out', type=float, default=0.2, help='Random seed.')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='Random seed.')
parser.add_argument('--weight_decay', type=float, default=1e-2, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50, help='Random seed.')
parser.add_argument('--batch_size', type=int, default=128, help='Random seed.')

parser.add_argument('--load_interaction', type=bool, default=True, help='Random seed.')
parser.add_argument('--device', type=str, default='cpu', help='Disables CUDA training.')
parser.add_argument('--u_idx', type=int, default=0, help='Random seed.') # 없는 index 면 error?
parser.add_argument('--r_idx', type=str, default='', help='Random seed.') # _ 로 나뉜거 ?
parser.add_argument('--topk', type=int, default=20, help='Random seed.')


def inference():
    args = parser.parse_args()        
    path = '/HGAT/recipe_data'

    recipe_ingredient = pd.read_csv(os.path.join(path, '레시피_재료_내용_raw.csv'))
    recipe_ingredient.dropna(subset=['재료_아이디'],inplace=True)

    train = pd.read_csv(os.path.join(path, 'train셋(73609개-220603_192931).csv'))
    test = pd.read_csv(os.path.join(path, 'test셋(4422개-220603_192931).csv'))

    if args.load_interaction:
        interaction = torch.load('/HGAT/data/interaction.pt').to(args.device)
    else:  
        data = pd.concat([train,test])
        user2idx = dataset.user2idx
        recipe2idx = dataset.recipe2idx
        interaction = torch.zeros(len(users), len(recipes)).to(args.device)

        for u in tqdm(data.유저_아이디.unique()):
            u_idx = user2idx[u]
            for r in data[data.유저_아이디==u].레시피_아이디.unique():
                r_idx = recipe2idx[r]
                interaction[u_idx][r_idx] = 1
        torch.save(interaction, '/HGAT/data/interaction.pt') 

    dataset = recipe_dataset( train,test, recipe_ingredient ,args.user_dimension)
    users, recipes, ings = dataset.get_user_recipe_ing()

    model = HGAT(
            user_dim = args.user_dimension, 
            dim_list = [args.recipe_dimension], 
            n_hidden_unit = args.hidden_unit, 
            r_hidden_unit = args.hidden_unit, 
            nclass = len(recipes), 
            n_dropout = args.drop_out, 
            r_dropout = args.drop_out, 
            alpha = args.alpha, 
            nheads = args.n_heads,
            device = args.device
            )     
    
    model.load_state_dict(torch.load('/HGAT/model_save/best_model.pt'))
    
    user_emb = dataset.user_embedding.weight.to(args.device)
    recipe_emb = dataset.recipe_embedding.weight.to(args.device)
    ing_emb = dataset.ing_embedding.weight.to(args.device)
    
    ################ 여기서 까지 띄워놓고  #######################3

    ############### 여기서 부터 반복해주세요 #############3    
    if args.u_idx > user_emb.shape[0]:
        print([])
        return torch.tensor([])    
    
    user_inter = interaction[args.u_idx]

    if args.r_idx == '':
        args.r_idx = []
    else:
        args.r_idx = list(map(int,args.r_idx.split('_')))
        user_inter[args.r_idx] = 1

    pred = model(user_emb[[args.u_idx]],[recipe_emb], [interaction[args.u_idx]])
    pred = F.log_softmax(pred, dim=0)
    pred[interaction[args.u_idx]>0] = -1e6
    
    print(pred.topk(args.topk))
    return pred.to('cpu').topk(args.topk)

if __name__ == '__main__':
    inference()