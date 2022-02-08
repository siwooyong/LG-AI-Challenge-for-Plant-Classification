# get model weights

model1_path = glob(opt['folder'] + '/fold1/*.bin')[-1]
model2_path = glob(opt['folder'] + '/fold2/*.bin')[-1]
model3_path = glob(opt['folder'] + '/fold3/*.bin')[-1]
model4_path = glob(opt['folder'] + '/fold4/*.bin')[-1]
model5_path = glob(opt['folder'] + '/fold5/*.bin')[-1]

# fold1 model
model1 = CustomModel(opt)
model1.load_state_dict(torch.load(model1_path, map_location='cpu'))
model1.to(device)
model1.eval()

# fold2 model
model2 = CustomModel(opt)
model2.load_state_dict(torch.load(model2_path, map_location='cpu'))
model2.to(device)
model2.eval()

# fold3 model
model3 = CustomModel(opt)
model3.load_state_dict(torch.load(model3_path, map_location='cpu'))
model3.to(device)
model3.eval()

# fold4 model
model4 = CustomModel(opt)
model4.load_state_dict(torch.load(model4_path, map_location='cpu'))
model4.to(device)
model4.eval()

# fold5 model
model5 = CustomModel(opt)
model5.load_state_dict(torch.load(model5_path, map_location='cpu'))
model5.to(device)
model5.eval()



# predict function

def predict(models, loader, mode):
    model1, model2, model3, model4, model5 = models

    preds = []
    for bi, data in enumerate(tqdm(loader)):
        data = [x.to(device) for x in data]
        if mode=='val':
            img, seq, label = data
        else:
            img, seq = data
        output1 = nn.Softmax(dim=-1)(model1(img, seq))
        output2 = nn.Softmax(dim=-1)(model2(img, seq))
        output3 = nn.Softmax(dim=-1)(model3(img, seq))
        output4 = nn.Softmax(dim=-1)(model4(img, seq))
        output5 = nn.Softmax(dim=-1)(model5(img, seq))

        output = output1 + output2 + output3 + output4 + output5
        pred = torch.argmax(output, dim=1).cpu().tolist()
        preds.extend(pred)
    return preds
  

# inference
models = [model1, model2, model3, model4, model5]

test_dataset = InferenceDataset(opt, test, mode='test')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt['batch_size'], num_workers=opt['worker_n'], shuffle=False)

with torch.no_grad():
    preds = predict(models, test_dataloader, mode='test')
