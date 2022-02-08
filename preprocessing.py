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
