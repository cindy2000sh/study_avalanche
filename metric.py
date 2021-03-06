import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def visualize(M, save_path, curr_time, strategy, partial=False):
    # heatmap
    plt.figure(figsize = (15,15))
    if not partial:
        ax = sns.heatmap(M, annot=True, fmt=".1%", cmap="YlGnBu")
        ax.set_title("Accuracy Matrix")
        plt.xlabel('Test Bucket')
        plt.ylabel('Train Bucket')
        plt.savefig(save_path+f'/figures/{curr_time}_{strategy}_iid.png')
    else:
        mask = np.zeros_like(M)
        mask[np.tril_indices_from(mask)] = True # lower triangle indices
        ax = sns.heatmap(M, annot=True, fmt=".1%", cmap="YlGnBu", mask = mask)
        ax.set_title("Accuracy Matrix")
        plt.xlabel('Test Bucket')
        plt.ylabel('Train Bucket')
        plt.savefig(save_path+f'/figures/{curr_time}_{strategy}_streaming.png')
    return

def in_domain(M):
    r,_ = M.shape
    return sum([M[i,i] for i in range(r)])/r

def next_domain(M):
    r,_ = M.shape
    return sum([M[i,i+1] for i in range(r-1)])/(r-1)

# TODO: check if exclude diagonals
def backward_transfer(M):
    r,_ = M.shape
    res = [M[i,j] for i in range(r) for j in range(i+1,r)]
    return sum(res)/len(res)

def forward_transfer(M):
    r,_ = M.shape
    res = [M[i,j] for i in range(r) for j in range(i)]
    return sum(res)/len(res)
