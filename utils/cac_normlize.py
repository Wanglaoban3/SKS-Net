import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from torchvision.transforms import ToPILImage
import torch
import transform
from dataset.data import SemData

all_data, all_label = [], []
benchmark = sys.argv[1]
if benchmark == 'KolektorSDD1':
    data_root = 'C:/wrd/KolektorSDD/data/KolektorSDD'
    inputWH = (704, 256)
elif benchmark == 'KolektorSDD2':
    data_root = 'C:/wrd/KolektorSDD2/data/KolektorSDD2'
    inputWH = (646, 230)
elif benchmark == 'CrackForst':
    data_root = 'C:/wrd/CrackForst'
    inputWH = (320, 480)
elif benchmark == 'RSDD1':
    data_root = 'C:/wrd/RSDDs/Type-I_RSDDs_dataset'
    inputWH = (1282, 160)
elif benchmark == 'RSDD2':
    data_root = 'C:/wrd/RSDDs/Type-II_RSDDs_dataset'
    inputWH = (1250, 55)
elif benchmark == 'Magnetic':
    data_root = 'C:/wrd/Magnetic'
    inputWH = (448, 448)

train_ann = 'train.txt'
test_ann = 'test.txt'
# mean = [138.9284, 133.7921, 129.6794]
# std = [18.7216, 18.2210, 20.5493]

trans = transform.Compose([
    transform.Resize(inputWH),
    transform.ToTensor(),
    # transform.Normalize(mean, std)
])

train_set = SemData(benchmark, 'train', data_root, train_ann, trans)
test_set = SemData(benchmark, 'test', data_root, test_ann, trans)

for x, label in train_set:
    all_data.append(x)
    all_label.append(label)
for x, label in test_set:
    all_data.append(x)
    all_label.append(label)
all_data = torch.stack(all_data)
all_label = torch.stack(all_label).float()
for i in range(3):
    print(all_data[:, i, :, :].mean(), all_data[:, i, :, :].std())
