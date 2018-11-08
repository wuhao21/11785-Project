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
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#hyper-parameter
num_class = 597
net = models.resnet152(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, num_class)
net = net.to(DEVICE)
batch_size = 20
epoches = 5
learning_rate = 1e-4
val = 0.2
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)


class OpenImageDataset(Dataset):

	def __init__(self, csvfile, root_dir, transform=None, test=False):
		csv = pd.read_csv(csvfile)
		csv = csv.loc[csv.ImageID.str.startswith('0')].head(600000)
		self.img_ids = csv.ImageID
		self.YMin = np.array(csv.YMin)
		self.YMax = np.array(csv.YMax)
		self.XMin = np.array(csv.XMin)
		self.XMax = np.array(csv.XMax)
		self.root_dir = root_dir
		if not test:
			self.n2l = dict(enumerate(list(csv.LabelName.unique())))
			self.l2n = dict((v,k) for k,v in self.n2l.items())
			self.labels = csv.LabelName
			self.num_class = csv.LabelName.nunique()
		self.transform = transform
		self.test = test

	def __len__(self):
		return len(self.img_ids)

	def __getitem__(self, idx):
		img_name = self.root_dir+self.img_ids[idx]+'.jpg'
		image = io.imread(img_name)
		#print('---------------------')
		if len(image.shape) == 1:
			image = image[0]
		if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
			image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
		if  (len(image.shape) == 3 and image.shape[2] == 4):
			image = cv2.cvtColor(image,cv2.COLOR_RGBA2RGB)

		h0, w0 = image.shape[:2]
		scale = 1024/max(h0, w0)
		h, w = int(round(h0*scale)), int(round(w0*scale))
		image = cv2.resize(image, (h, w), interpolation = cv2.INTER_AREA)
		image = image[
		    max(0, int(self.YMin[idx]*w)-2):min(w-1, int(self.YMax[idx]*w)+2),
		    max(0, int(self.XMin[idx]*h)-2):min(h-1, int(self.XMax[idx]*h)+2)
		]
		image = np.transpose(image, (2, 0, 1))

		#print(image.shape)
		if self.transform:
			try:
				this_shape = image.shape
				image = self.transform(np.array(image))
			except ValueError:
				print(this_shape)
				print(img_name)
				print(max(0, int(self.YMin[idx]*w)-1), min(w-1, int(self.YMax[idx]*w)+1), max(0, int(self.XMin[idx]*h)-1), min(h-1, int(self.XMax[idx]*h)+1))
				print(h, w)
				exit(0)
			image = image[0:3]
		#print(image.size())
		if self.test:
			return image
		else:
			return (image, self.l2n[self.labels[idx]])

def get_dataloader():
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
	train_dataset = OpenImageDataset(
		csvfile='all/train_bounding_boxes.csv',
		root_dir='train_0/',
		transform=pre_transform,
		test=False,
	)
	'''
	test_dataset = OpenImageDataset(
		csvfile='hw7data/test.csv',
		root_dir='hw7data/images/',
		transform=pre_transform,
		test=True,
	)
	'''
	train_dataloader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
	)
	'''
	test_dataloader = DataLoader(
		test_dataset,
		batch_size=batch_size,
		shuffle=False,
	)
	'''
	#return train_dataloader, test_dataloader
	return train_dataloader

def run():
	print('Get DataLoader...')
	train_dataloader = get_dataloader()
	print('Start training...')
	tls = []
	tas = []
	vls = []
	vas = []
	for epoch in range(epoches):
		train_loss = 0.0
		train_acc = 0.0
		train_size = 0
		val_loss = 0.0
		val_acc = 0.0
		val_size = 0
		cnt = 0
		size = len(train_dataloader)
		val_bound = int(size - size * val)
		net.train()
		for X, y in tqdm(train_dataloader, ascii=True):
			cnt += 1
			X = X.to(DEVICE)
			y = y.to(DEVICE)
			optimizer.zero_grad()
			out = net(X)
			preds = torch.max(out, 1)[1]
			loss = criterion(out, y)
			if cnt < val_bound:
				loss.backward()
				optimizer.step()
				train_size += len(X)
				train_loss += loss.item()
				train_acc += torch.sum(preds == y.data).item() * 1.0
			else:
				val_size += len(X)
				val_loss += loss.item()
				val_acc += torch.sum(preds == y.data).item() * 1.0
		train_loss /= train_size
		train_acc /= train_size
		val_loss /= val_size
		val_acc /= val_size
		print("Epoch:%d:"%epoch)
		print("train loss:", train_loss, "train accuracy:", train_acc)
		print("validation loss:", val_loss, "validation accuracy:", val_acc)
		tls.append(train_loss)
		tas.append(train_acc)
		vls.append(val_loss)
		vas.append(val_acc)
		'''
		res = []
		net.eval()
		for X in tqdm(test_dataloader, ascii=True):
			X = X.to(DEVICE)
			out = net(X)
			preds = torch.max(out, 1)[1]
			res.extend(preds)
		with open('result%d.txt'%epoch, 'w') as f:
			f.write("landmark_id\n")
			for p in res:
				f.write("%d\n"%p)
		'''
		torch.save(net.state_dict(), 'tmp/epoch%d'%(epoch))
	x = np.arange(epoches)
	plt.plot(x, tls, 'x-', label="train")
	plt.plot(x, vls, 'o-', label="validation")
	plt.legend()
	plt.show()
if __name__ == '__main__':
	run()