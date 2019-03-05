import torch as th
from model import LBNet_1
import time
from dataset import DatasetForTest, loadImage
from torch.utils.data import DataLoader
import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('snapshot', help='Path to the snapshot to load and test')
parser.add_argument('--test_data_path', default='../data/gei/', help='Path to test data to run test on')
args = parser.parse_args()
print(args)

val_acc = None
lbnet = LBNet_1()
device = th.device("cuda:0")
checkpoint = th.load(args.snapshot)
lbnet = lbnet.to(device)
lbnet.load_state_dict(checkpoint['model'])
lbnet.eval()

bs = 128
testset = DatasetForTest(args.test_data_path)
testset = DataLoader(testset, bs, num_workers=8)
print('Start Testing...')
localtime = time.asctime(time.localtime(time.time()))
print('Evaluation starts at {}'.format(localtime))
sim = np.zeros((50, 50, 2, 4, 11, 11))  # probe person,
# gallery person, probe set, gallery set, probe angle, gallery_angle

for iteration, data in enumerate(testset):
    sys.stdout.write('iteration: {}\r'.format(iteration))
    sys.stdout.flush()
    img1, img2, label, (ids1, ids2), p_conds, p_angles, g_conds, g_angles = data
    img1 = img1.to(device).to(th.float32)
    img2 = img2.to(device).to(th.float32)
    img = th.cat((img1, img2), 1)
    output = lbnet(img)
    for ix in range(len(ids1)):
        id1 = ids1[ix] - 75
        p_angle = p_angles[ix]
        p_set = p_conds[ix]
        id2 = ids2[ix] - 75
        g_angle = g_angles[ix]
        g_set = g_conds[ix]
        sim[id1][id2][p_set][g_set][p_angle][g_angle] = output[ix].cpu().item()

localtime = time.asctime(time.localtime(time.time()))
print('Evaluation ends at {}'.format(localtime))

np.save('similarity.npy', sim)
print('Saved similarities into similarity.npy')
