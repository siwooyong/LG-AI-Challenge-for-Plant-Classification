
# inference dataset

def valTransform():
  return Compose([
                  Resize(opt['img_size1'], opt['img_size2']),
                  Normalize(mean=opt['mean'], std=opt['std'], max_pixel_value=255.0, p=1.0),
                  ToTensorV2(p=1.0),
              ], p=1.)
  
class InferenceDataset(Dataset):
    def __init__(self, opt, files, mode):
        self.files = files
        self.mode = mode
        self.csv_check = [0]*len(self.files)
        self.seq = [None]*len(self.files)
        self.minmax_dict = opt['minmax_dict']
        self.max_len = opt['max_len']
        self.label_encoder = opt['label_dict']

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        file = self.files[i]
        file_name = file.split('/')[-1]
        
        if self.csv_check[i] == 0:
            csv_path = f'{file}/{file_name}.csv'
            df = pd.read_csv(csv_path)
            try:
                estiTime1, estiTime2 = df.iloc[0]['측정시각'], df.iloc[1]['측정시각']
            except:
                estiTime1, estiTime2 = 0, 1

            df = df[self.minmax_dict.keys()]
            df = df.replace('-', 0)
            
            if estiTime1==estiTime2 and len(df)>400:
                df = df[0::2].reset_index(drop=True)
                
            
            # minmax-scaling
            for col in df.columns:
                df[col] = df[col].astype(float) - self.minmax_dict[col][0]
                df[col] = df[col] / (self.minmax_dict[col][1]-self.minmax_dict[col][0])

            # zero-padding
            pad = np.zeros((self.max_len, len(df.columns)))
            length = min(self.max_len, len(df))
            pad[-length:] = df.to_numpy()[-length:]

            # transpose-to-sequential-data
            seq = torch.tensor(pad, dtype=torch.float32)
            self.seq[i] = seq
            self.csv_check[i] = 1
        else:
            seq = self.seq[i]
        
        image_path = f'{file}/{file_name}.jpg'
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
        img = valTransform()(image=img)['image'] 

        if self.mode == 'val':
            json_path = f'{file}/{file_name}.json'
            with open(json_path, 'r') as f:
                json_file = json.load(f)
            
            crop = json_file['annotations']['crop']
            disease = json_file['annotations']['disease']
            risk = json_file['annotations']['risk']
            label = torch.tensor(self.label_encoder[f'{crop}_{disease}_{risk}'], dtype=torch.long)
            
            return img, seq, label
        else:
            return img, seq
