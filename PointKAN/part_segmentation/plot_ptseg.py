"""
Plot the parts.
python plot_ptseg.py --model model31G --exp_name demo1 --id 1
"""
from __future__ import print_function
import os
import argparse
import torch
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from util.data_util import PartNormalDataset
import torch.nn.functional as F
import torch.nn as nn
import model as models
import numpy as np
from torch.utils.data import DataLoader
from util.util import to_categorical, compute_overall_iou, IOStream
from tqdm import tqdm
from collections import defaultdict
from torch.autograd import Variable
import random

# import matplotlib.colors as mcolors
# def_colors = mcolors.CSS4_COLORS
# colrs_list = []
# np.random.seed(2021)
# for k, v in def_colors.items():
#     colrs_list.append(k)
# np.random.shuffle(colrs_list)
colrs_list = [
    "C0", "C1","C2","C3","C4","C5","C6","C7","C8","C9","deepskyblue", "m","deeppink","hotpink","lime","c","y",
    "gold","darkorange","g","orangered","tomato","tan","darkorchid","violet","C0", "C1","C2","C3","C4","C5","C6","C7","C8","C9","deepskyblue", "m","deeppink","hotpink","lime","c","y",
    "gold","darkorange","g","orangered","tomato","tan","darkorchid","violet","C0", "C1","C2","C3","C4","C5","C6","C7","C8","C9","deepskyblue", "m","deeppink","hotpink","lime","c","y",
    "gold","darkorange","g","orangered","tomato","tan","darkorchid","violet"
]

def test(args):
    # Dataloader
    test_data = PartNormalDataset(npoints=2048, split='test', normalize=False)
    print("===> The number of test data is:%d", len(test_data))
    # Try to load models
    print("===> Create model...")
    num_part = 50
    device = torch.device("cuda" if args.cuda else "cpu")
    model = models.__dict__[args.model](num_part).to(device)
    print("===> Load checkpoint...")
    from collections import OrderedDict
    state_dict = torch.load("checkpoints/%s/best_%s_model.pth" % (args.exp_name, args.model_type),
                            map_location=torch.device('cpu'))['model']
    new_state_dict = OrderedDict()
    for layer in state_dict:
        new_state_dict[layer.replace('module.', '')] = state_dict[layer]
    model.load_state_dict(new_state_dict)
    print("===> Start evaluate...")
    model.eval()
    num_classes = 16
    for i in range(args.id, args.id+1): 
        points, label, target, norm_plt = test_data.__getitem__(i)
        points = torch.tensor(points).unsqueeze(dim=0)
        label = torch.tensor(label).unsqueeze(dim=0)
        target = torch.tensor(target).unsqueeze(dim=0)
        norm_plt = torch.tensor(norm_plt).unsqueeze(dim=0)
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(dim=0).cuda(
            non_blocking=True), target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
        with torch.no_grad():
                cls_lable = to_categorical(label, num_classes)
                predict = model(points, norm_plt, cls_lable)  # b,n,50
        # up to now, points [1, 3, 2048]  predict [1, 2048, 50] target [1, 2048]
        predict = predict.max(dim=-1)[1]
        predict = predict.squeeze(dim=0).cpu().data.numpy()  # 2048
        target = target.squeeze(dim=0).cpu().data.numpy()   # 2048
        points = points.transpose(2, 1).squeeze(dim=0).cpu().data.numpy() #[2048,3]

        np.savetxt(f"figures/{i}-point.txt", points)
        np.savetxt(f"figures/{i}-target.txt", target)
        np.savetxt(f"figures/{i}-predict.txt", predict)

        # start plot
        print(f"===> stat plotting")
        plot_xyz(points, target, name=f"figures/{i}-gt.pdf")
        plot_xyz(points, predict, name=f"figures/{i}-predict.pdf")


def plot_xyz(xyz, target, name="figures/figure.pdf"):
    fig = pyplot.figure(figsize=(8, 6))
    ax = Axes3D(fig)
    ax.view_init(elev=30, azim=45)
    # ax = fig.gca(projection='3d')
    x_vals = xyz[:, 0]
    y_vals = xyz[:, 1]
    z_vals = xyz[:, 2]
    ax.set_xlim3d(min(x_vals)*0.9, max(x_vals)*0.9)
    ax.set_ylim3d(min(y_vals)*0.9, max(y_vals)*0.9)
    ax.set_zlim3d(min(z_vals)*0.9, max(z_vals)*0.9)
    for i in range(0,2048):
        col = int(target[i])
        ax.scatter(x_vals[i], y_vals[i], z_vals[i], c=colrs_list[col], marker="o", s=30, alpha=0.7)
    ax.set_axis_off()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    # pyplot.tight_layout()
    fig.savefig(name, bbox_inches='tight', pad_inches=0.1, transparent=True)
    pyplot.close()

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='3D Shape Part Segmentation')
    parser.add_argument('--model', type=str, default='PointMLP1')
    parser.add_argument('--id', type=int, default='679')
    parser.add_argument('--exp_name', type=str, default='demo3-250406', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--model_type', type=str, default='insiou',
                        help='choose to test the best insiou/clsiou/acc model (options: insiou, clsiou, acc)')

    args = parser.parse_args()
    args.exp_name = args.model+"_"+args.exp_name
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    test(args)