import torch

def recall_k(prediction,labels, topk):
    recall_at_k = 0
    for i in range(prediction.shape[0]):
        r_i = prediction[i].topk(topk).indices.tolist()
        l_i = torch.where(labels[i] == 1)[0].tolist()
        recall_i = len(set(r_i) & set(l_i)) / min(topk,len(l_i))
        recall_at_k += recall_i
    recall_at_k = recall_at_k/prediction.shape[0]
    return round(recall_at_k * 100,4)
