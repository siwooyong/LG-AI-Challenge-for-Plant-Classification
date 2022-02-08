# model

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.model = timm.create_model(model_name=opt['enc_name'], 
                                       pretrained=opt['vision_pretrain'], 
                                       num_classes=0)
    
    def forward(self, inputs):
        output = self.model(inputs)
        return output


class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.decoder = nn.GRU(opt['feature_n'], opt['embedding_dim'], bidirectional = opt['bidirectional'])
        self.dense = nn.Linear(2*opt['max_len']*opt['embedding_dim'], opt['dec_dim'])
        
        self.f1 = nn.Linear(opt['enc_dim']+opt['dec_dim'], opt['enc_dim']+opt['dec_dim'])
        self.out = nn.Linear(opt['enc_dim']+opt['dec_dim'], opt['class_n'])
        self.dropout = nn.Dropout(opt['dropout_rate'])
        self.relu = nn.ReLU()

    def init_weight(self):
        torch.nn.init.xavier_uniform_(self.f1.weight)  
        torch.nn.init.xavier_uniform_(self.dense.weight)  
        torch.nn.init.xavier_uniform_(self.out.weight)  


    def forward(self, enc_out, dec_inp):
        dec_out, _ = self.decoder(dec_inp)
        dec_out = self.dense(dec_out.view(dec_out.size(0), -1))

        concat = torch.cat([enc_out, dec_out], dim=1) 
        concat = self.f1(self.relu(concat))
        concat = self.dropout(self.relu(concat))
        output = self.out(concat)
        return output


class CustomModel(nn.Module):
    def __init__(self, opt):
        super(CustomModel, self).__init__()
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)
        self.to(device)
        
    def forward(self, img, seq):
        enc_out = self.encoder(img)
        output = self.decoder(enc_out, seq)
        
        return output
