import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io
import matplotlib.pyplot as plt
import cv2
from finetune import OpenImageDataset, get_dataloader

def f2_score(res, tar, eps=1e-9, beta=2):
	TP = 0
	FP = 0
	FN = 0
	for x in res:
		if x in tar:
			TP += 1
		else:
			FP += 1
	for y in tar:
		if y not in res:
			FN += 1
	precision = TP * 1.0 / (TP + FP + eps)
	recall = TP * 1.0 / (TP + FN + eps)
	F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + eps)
	return F2

num_class = 597
net = models.resnet18()
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, num_class)
net.load_state_dict(torch.load('tmp/epoch1'))
net.eval()

print('Get DataLoader...')
train_dataloader = get_dataloader()
n2l = train_dataloader.dataset.n2l
normalize = transforms.Normalize(
	mean=[0.485, 0.456, 0.406], 
	std=[0.229, 0.224, 0.225],
)
pre_transform = transforms.Compose([
	transforms.ToPILImage(),
	transforms.RandomResizedCrop(224),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	normalize,
])

csv = pd.read_csv('all/tuning_labels.csv')
total = 0.0
for index, row in tqdm(csv.iterrows(), ascii=True):
	img_name = 'all/stage_1_test_images/'+row[0]+'.jpg'
	image = io.imread(img_name)
	if len(image.shape) == 1:
		image = image[0]
	if len(image.shape) == 2:
		image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
	image = np.transpose(image, (2, 0, 1))
	if image.shape[0] == 4:
		image = cv2.cvtColor(image,cv2.COLOR_RGBA2RGB)
	image = pre_transform(np.array(image))
	out = net(image.unsqueeze(0))
	res = [n2l[x] for x in out.detach().numpy()[0].argsort()[-3:][::-1]]
	tar = row[1].split()
	total += f2_score(res, tar)
	print ("index: ", index, "score: ", total/float(index+1))
print(total/1000.0)