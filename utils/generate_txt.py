import os
import sys

# benchmark = sys.argv[1]
benchmark = 'NEUSEG'
# root_path = sys.argv[2]
# root_path = 'C:/wrd/KolektorSDD2/data/KolektorSDD2'
root_path = 'C:/wrd/NEU_Seg'
train_path = 'train.txt'
test_path = 'test.txt'

def generate_kolektorSDD1(root_path, split, out_path):
    if split == 'train':
        dir_path = os.path.join(root_path, 'Train')
        root_dir = 'Train'
    else:
        dir_path = os.path.join(root_path, 'Test')
        root_dir = 'Test'

    filenames = []
    dirs = os.listdir(dir_path)
    for sub_dir in dirs:
        sub_dirs = os.path.join(dir_path, sub_dir)
        if not os.path.isdir(sub_dirs):
            continue
        for subsub in os.listdir(sub_dirs):
            if "label" in subsub:
                continue
            filenames.append(os.path.join(root_dir, sub_dir, subsub))
    f = open(os.path.join(root_path, out_path), 'w')
    for filename in filenames:
        labelname = filename.split('.jpg')[0]+'_label.bmp'
        f.write(f"{filename}\t{labelname}\n")
    f.close()
    return

def generate_kolektorSDD2(root_path, split, out_path):
    out_path = os.path.join(root_path, out_path)
    if split == 'train':
        dir_path = os.path.join(root_path, 'train')
    else:
        dir_path = os.path.join(root_path, 'test')
    filenames = []
    dirs = os.listdir(dir_path)
    for file in dirs:
        if 'label' in file or 'txt' in file or 'GT' in file or 'copy' in file:
            continue
        filenames.append(os.path.join(split, file))
    f = open(out_path, 'w')
    for filename in filenames:
        labelname = filename.split('.png')[0]+'_GT.png'
        f.write(f'{filename}\t{labelname}\n')
    f.close()
    return

def generate_CrackForst(root_path):
    import random
    random.seed(107)
    ratio = 0.8
    img_path = os.path.join(root_path, 'Images')
    img_paths = []
    for img in os.listdir(img_path):
        img_paths.append(img)
    random.shuffle(img_paths)
    split_num = int(ratio * len(img_paths))
    train_imgs, test_imgs = img_paths[:split_num], img_paths[split_num:]
    f = open(os.path.join(root_path, 'train.txt'), 'w')
    for train_img in train_imgs:
        label = train_img.split('.jpg')[0]+'_label.PNG'
        f.write(f'Images/{train_img}\tMasks/{label}\n')
    f.close()
    f = open(os.path.join(root_path, 'test.txt'), 'w')
    for test_img in test_imgs:
        label = test_img.split('.jpg')[0] + '_label.PNG'
        f.write(f'Images/{test_img}\tMasks/{label}\n')
    f.close()

def generate_RSDD1(root_path):
    import random
    random.seed(107)
    ratio = 0.8
    img_path = os.path.join(root_path, 'Rail surface images')
    img_paths = []
    for img in os.listdir(img_path):
        img_paths.append(img)
    random.shuffle(img_paths)
    split_num = int(ratio * len(img_paths))
    train_imgs, test_imgs = img_paths[:split_num], img_paths[split_num:]
    f = open(os.path.join(root_path, 'train.txt'), 'w')
    for train_img in train_imgs:
        label = train_img
        f.write(f'Rail surface images/{train_img}\tGroundTruth/{label}\n')
    f.close()
    f = open(os.path.join(root_path, 'test.txt'), 'w')
    for test_img in test_imgs:
        label = test_img
        f.write(f'Rail surface images/{test_img}\tGroundTruth/{label}\n')
    f.close()


def generate_Magnetic(root_path):
    img_paths = []
    for path in os.listdir(root_path):
        for sub_path in os.listdir(os.path.join(root_path, path)):
            for sub_sub_path in os.listdir(os.path.join(root_path, path, sub_path)):
                if '.jpg' in sub_sub_path:
                    img_paths.append(os.path.join(path, sub_path, sub_sub_path).replace('\\', '/'))
    import random
    random.seed(107)
    ratio = 0.8
    random.shuffle(img_paths)
    split_num = int(ratio * len(img_paths))
    train_imgs, test_imgs = img_paths[:split_num], img_paths[split_num:]
    f = open(os.path.join(root_path, 'train.txt'), 'w')
    for train_img in train_imgs:
        label = train_img.split('.jpg')[0]+'.png'
        f.write(f'{train_img} {label}\n')
    f.close()
    f = open(os.path.join(root_path, 'test.txt'), 'w')
    for test_img in test_imgs:
        label = test_img.split('.jpg')[0] + '.png'
        f.write(f'{test_img} {label}\n')
    f.close()

def generate_neuseg(root_path, split, out_path):
    out_path = os.path.join(root_path, out_path)
    if split == 'train':
        dir_path = os.path.join(root_path, 'images/training')
    else:
        dir_path = os.path.join(root_path, 'images/test')
    filenames = []
    dirs = os.listdir(dir_path)
    for file in dirs:
        if '.jpg' in file:
            filenames.append(os.path.join(dir_path, file))
    f = open(out_path, 'w')
    for filename in filenames:
        labelname = filename.replace('images', 'annotations').replace('jpg', 'png')
        f.write(f'{filename}\t{labelname}\n')
    f.close()
    return

if benchmark == 'kolektorSDD1':
    generate_kolektorSDD1(root_path, 'train', train_path)
    generate_kolektorSDD1(root_path, 'test', test_path)
elif benchmark == 'kolektorSDD2':
    generate_kolektorSDD2(root_path, 'train', train_path)
    generate_kolektorSDD2(root_path, 'test', test_path)
elif benchmark == 'CrackForst':
    generate_CrackForst(root_path)
elif benchmark == 'RSDD1':
    generate_RSDD1(root_path)
elif benchmark == 'RSDD2':
    generate_RSDD1(root_path)
elif benchmark == 'Magnetic':
    generate_Magnetic(root_path)
elif benchmark == 'NEUSEG':
    generate_neuseg(root_path, 'train', train_path)
    generate_neuseg(root_path, 'test', test_path)