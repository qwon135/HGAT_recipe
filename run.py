import pandas as pd
import os
from models import HGAT
import torch
from torch import nn
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
from dataset import recipe_dataset
from utils import recall_k
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
parser.add_argument('--topk', type=int, default=100, help='Random seed.')
parser.add_argument('--load_label', type=bool, default=True, help='Random seed.')
parser.add_argument('--device', type=str, default='cuda', help='Disables CUDA training.')

def recall_k(prediction,labels, topk):
    recall_at_k = 0
    for i in range(prediction.shape[0]):
        r_i = prediction[i].topk(topk).indices.tolist()
        l_i = torch.where(labels[i] == 1)[0].tolist()
        recall_i = len(set(r_i) & set(l_i)) / min(topk,len(l_i))
        recall_at_k += recall_i
    recall_at_k = recall_at_k/prediction.shape[0]
    return round(recall_at_k * 100,4)


def run():
    args = parser.parse_args()

    path = '/HGAT/recipe_data'

    recipe_ingredient = pd.read_csv(os.path.join(path, '레시피_재료_내용_raw.csv'))
    recipe_ingredient.dropna(subset=['재료_아이디'],inplace=True)

    train = pd.read_csv(os.path.join(path, 'train셋(73609개-220603_192931).csv'))
    test = pd.read_csv(os.path.join(path, 'test셋(4422개-220603_192931).csv'))

    dataset = recipe_dataset( train,test, recipe_ingredient, u_dim = args.user_dimension)

    users, recipes, ings = dataset.get_user_recipe_ing()

    user_rel_matrix, recipe_rel_matrix = torch.load('/HGAT/data/1.pt'),[  
                                                                            torch.load('/HGAT/data/2.pt'), 
                                                                            torch.load('/HGAT/data/3.pt'), 
                                                                            torch.load('/HGAT/data/4.pt')]

    user_emb = dataset.user_embedding.weight.to(args.device)
    recipe_emb = dataset.recipe_embedding.weight.to(args.device)
    ing_emb = dataset.ing_embedding.weight.to(args.device)

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

    epochs = args.epochs
    topk = args.topk

    optimizer1 = optim.Adam(model.parameters(), 
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss(reduction="mean")

    urm = user_rel_matrix.to(args.device)
    for i in range(len(recipe_rel_matrix)):
        recipe_rel_matrix[i] = recipe_rel_matrix[i].to(args.device)

    dataloader = BatchSampler(RandomSampler(user_emb), args.batch_size, drop_last=False)

    if args.load_label:
        label = torch.load('/HGAT/data/label.pt').to(args.device)
    else:        
        user2idx = dataset.user2idx
        recipe2idx = dataset.recipe2idx
        label = torch.zeros(len(users), len(recipes)).to(args.device)
        for u in tqdm(train.유저_아이디.unique()):
            u_idx = user2idx[u]
            for r in test[test.유저_아이디==u].레시피_아이디.unique():
                r_idx = recipe2idx[r]
                label[u_idx][r_idx] = 1
        torch.save(label, '/HGAT/data/label.pt') 

    best_loss = 1e6
    best_recall = 0
    for e in tqdm(range(epochs)):
        for b_idx in dataloader:
            model.train()        
            
            h_u = model(user_emb[b_idx], [recipe_emb], [urm[b_idx]])                
            train_pred = F.log_softmax(h_u, dim=1)

            optimizer1.zero_grad()
                    
            loss_train = criterion(train_pred, urm[b_idx])  
            loss_train.backward()
            
            optimizer1.step()                
            if loss_train.detach().item() < best_loss:
                torch.save(model.state_dict(), '/HGAT/model_save/best_model.pt')
                print('save best model!')
                best_loss = loss_train.clone().detach().item()                                                            
        
        with torch.no_grad():
            model.eval()        
            h_u = model(user_emb, [recipe_emb], [urm])        
            
            pred = F.log_softmax(h_u, dim=1)
            pred[torch.where(urm > 0)] = -1e6
            
            rck = recall_k(pred.squeeze(1), label, topk)
            if rck > best_recall:
                best_recall = rck
            print(f"recall@{topk} :{rck}%")
            print(f'epochs : {e}, train, loss : {loss_train}')    
            print('---------------------------------------------------------------------')
    print(f"best_reacll@{topk} : {best_recall}%")
    print(f"best_loss : {best_loss}%")
if __name__ == '__main__':
    run()