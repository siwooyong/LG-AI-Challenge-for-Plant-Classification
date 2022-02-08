# train function

class CustomTrainer:
    def __init__(self, model, folder, fold):
        self.model=model
        
        self.save_dir = f'/content/{folder}'
        if not os.path.exists(self.save_dir):
          os.makedirs(self.save_dir)
          
        self.optimizer = AdamW(model.parameters(), lr=opt['lr'])
        self.scaler = torch.cuda.amp.GradScaler() 

        total_steps = int(len(train_dataset)*opt['epoch_n']/(opt['batch_size']))
        warmup_steps = 1149
        print('total_steps: ', total_steps)
        print('warmup_steps: ', warmup_steps)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        self.loss_fn = nn.CrossEntropyLoss()
        self.val_loss_fn = nn.CrossEntropyLoss()

        self.best_score = 0.0


    def run(self, train_loader, val_loader):
        for epoch in range(opt['epoch_n']):
            gc.collect()
            learning_rate = self.optimizer.param_groups[0]['lr']
            print('learning_rate: ', learning_rate)
            print(f'----- train, epoch{epoch+1} -----')
            train_loss, train_score = self.train_function(train_loader, epoch)
            print(' ')
            print(f'train_loss: {train_loss:.6f}, train_score: {train_score:.6f}')

            print('----------------------------------')

            print(f'----- val, epoch{epoch+1} -----')
            with torch.no_grad():
                val_loss, val_score = self.val_function(val_loader)
            print(' ')
            print(f'val_loss: {val_loss:.6f}, val_score: {val_score:.6f}')


            if epoch+1 >= 16 and val_score >= self.best_score:
                torch.save(self.model.state_dict(), self.save_dir+f"/best-acc-epoch{epoch+1}.bin")
                self.best_score=val_score
                print(f'model is saved when epoch is : {epoch+1}')


    def train_function(self, train_loader, epoch):
        self.model.train()

        total_loss = 0.0
        total_score = 0.0
        for bi, data in enumerate(tqdm(train_loader)):
            data = [x.to(device) for x in data]
            img, seq, label = data

            self.optimizer.zero_grad()
            
            # use mix or not
            if opt['mix']!=None and epoch < opt['epoch_n']-10: 
                mix_decision = np.random.rand()
                if opt['mix'] == 'cutmix' and mix_decision < opt['mix_prob']:
                    img, mix_labels = cutmix(img, label, 1.0)
                else: 
                  pass
            else: mix_decision = 1

            if opt['epoch_n']-10 <= epoch:
                assert mix_decision == 1
            
            # use amp or not
            if opt['precision'] == 'float':
                out = self.model(img, seq)

                if mix_decision < opt['mix_prob']:
                    loss = self.loss_fn(out, mix_labels[0])*mix_labels[2] + self.loss_fn(out, mix_labels[1])*(1-mix_labels[2])
                else:
                    loss = self.loss_fn(out, label)

                loss.backward()
                self.optimizer.step()
            else: 
                with torch.cuda.amp.autocast():
                    out = self.model(img, seq)
                    if mix_decision < opt['mix_prob']:
                        loss = self.loss_fn(out, mix_labels[0])*mix_labels[2] + self.loss_fn(out, mix_labels[1])*(1-mix_labels[2])
                    else:
                        loss = self.loss_fn(out, label)

                self.scaler.scale(loss).backward()  
                self.scaler.step(self.optimizer) 
                self.scaler.update()              
            
            self.scheduler.step()
            total_loss+=loss.detach().cpu()

            total_score+=f1_score(label.cpu(), out.argmax(1).cpu(), average='macro')
        return total_loss/len(train_loader), total_score/len(train_loader)

    def val_function(self, val_loader):
        self.model.eval()

        total_loss = 0.0
        preds, targets = [], []
        for bi, data in enumerate(tqdm(val_loader)):
            data = [x.to(device) for x in data]
            img, seq, label = data

            out = self.model(img, seq)
            loss = self.val_loss_fn(out, label)

            total_loss+=loss.detach().cpu()

            pred = out.argmax(1).detach().cpu().tolist()
            target = label.reshape(-1).detach().cpu().tolist()

            preds.extend(pred)
            targets.extend(target)
        
        score = f1_score(targets, preds, average='macro')
        return total_loss/len(val_loader), score

    def log(self, message):
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
