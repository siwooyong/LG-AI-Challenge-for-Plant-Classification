# get sequence feature information

csv_feature_dict = {'내부 습도 1 최고': [25.9, 100.0],
                    '내부 습도 1 최저': [0.0, 100.0],
                    '내부 습도 1 평균': [23.7, 100.0],
                    '내부 온도 1 최고': [3.4, 47.6],
                    '내부 온도 1 최저': [3.3, 47.0],
                    '내부 온도 1 평균': [3.4, 47.3],
                    '내부 이슬점 최고': [0.2, 34.7],
                    '내부 이슬점 최저': [0.0, 34.4],
                    '내부 이슬점 평균': [0.1, 34.5]}


# label encoder, decoder 

train_json = sorted(glob('train/*/*.json'))

labels = []
for i in range(len(train_json)):
    with open(train_json[i], 'r') as f:
        sample = json.load(f)
        crop = sample['annotations']['crop']
        disease = sample['annotations']['disease']
        risk = sample['annotations']['risk']
        label=f"{crop}_{disease}_{risk}"
        labels.append(label)

label_encoder = sorted(np.unique(labels))
label_encoder = {key:value for key,value in zip(label_encoder, range(len(label_encoder)))}
label_decoder = {val:key for key, val in label_encoder.items()}


# cutmix function

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data, targets
