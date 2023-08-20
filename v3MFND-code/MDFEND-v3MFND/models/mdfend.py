import os
import torch
import tqdm
import torch.nn as nn
import numpy as np
from .layers import *
from sklearn.metrics import *
from transformers import BertModel, AutoModel
from utils.utils import data2gpu, Averager, metrics, Recorder
import torch.nn as nn
from torchsummary import summary   
import torchvision.models as models

 

class MultiDomainFENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, bert, dropout, emb_type , fusion_source, fusion_type):
        super(MultiDomainFENDModel, self).__init__()
        self.domain_num = 10
        self.gamma = 10
        self.num_expert = 6
        self.fea_size =256
        self.emb_type = emb_type
        self.fusion_source = fusion_source
        self.fusion_type = fusion_type
        
        if(emb_type == 'bert'):
            # self.bert = BertModel.from_pretrained(bert).requires_grad_(False)
            self.bert = AutoModel.from_pretrained("vinai/phobert-base").requires_grad_(False)
        
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        expert = []
        for i in range(self.num_expert):
            expert.append(cnn_extractor(feature_kernel, emb_dim))
        self.expert = nn.ModuleList(expert)

        self.gate = nn.Sequential(nn.Linear(emb_dim * 2, mlp_dims[-1]),
                                      nn.ReLU(),
                                      nn.Linear(mlp_dims[-1], self.num_expert),
                                      nn.Softmax(dim = 1))

        self.attention = MaskAttention(emb_dim)

        self.domain_embedder = nn.Embedding(num_embeddings = self.domain_num, embedding_dim = emb_dim)
        self.specific_extractor = SelfAttentionFeatureExtract(multi_head_num = 1, input_size=emb_dim, output_size=self.fea_size)
        self.resize_one = nn.Sequential(
                                        torch.nn.Linear(320, 1)
                                      )
        self.norm_text = nn.Sequential(
                                        # nn.Linear(320, 320),
                                        nn.BatchNorm1d(320), 
                                        nn.ReLU(), 
                                      nn.Dropout(p=0.4)
                                    
                                      )
        ## 17 - 320
        self.resize_meta =  nn.Sequential(nn.Linear(17, 320),
                                        nn.BatchNorm1d(320),
                                      nn.ReLU(), 
                                      nn.Dropout(p=0.4)
                                      )
        
        ## 18 -> 320
        self.resize_emo =  nn.Sequential(nn.Linear(18, 320),
                                        nn.BatchNorm1d(320),
                                      nn.ReLU(), 
                                      nn.Dropout(p=0.4)
                                      )
        ## 4096 - 2742 - 320:
        self.resize_img2 =  nn.Sequential(nn.Linear(4096, 2742),
                                        nn.BatchNorm1d(2742),
                                      nn.ReLU(), 
                                      nn.Dropout(p=0.4), 
                                      
                                      nn.Linear( 2742, 320),
                                        nn.BatchNorm1d(320),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.4), 
                                      )
        ## 512 -> 320
        self.resize_img =  nn.Sequential(nn.Linear(512, 320),
                                        nn.BatchNorm1d(320),
                                      nn.ReLU(), 
                                      nn.Dropout(p=0.4)
                                      
                       
                                      )
                                      

        ## img dim - 4096, text - 320, meta - 17
        if (self.fusion_source == 0):
            self.classifier = MLP( 320, mlp_dims, dropout) #text
        if (self.fusion_source == 1):
            if self.fusion_type == "concat":
                self.classifier = MLP( 640, mlp_dims, dropout) #text + img: concat
            else: 
                self.classifier = MLP( 320, mlp_dims, dropout) #text + img: mean, sum, weighted sum
        if (self.fusion_source == 2):
            self.classifier = MLP(960, mlp_dims, dropout) #text+ img + meta
            # self.classifier = MLP( 320, mlp_dims, dropout) #text + img + meta: mean, sum, weighted sum
            

        
        if (self.fusion_source == 3):
            # self.classifier = MLP(640, mlp_dims, dropout) #text+meta:concat
            self.classifier = MLP( 320, mlp_dims, dropout) #text + meta: mean, sum, weighted sum
        if (self.fusion_source == 4):
            if self.fusion_type == "concat":   
                self.classifier = MLP( 640, mlp_dims, dropout) # text + emo: concat
            else:
                self.classifier = MLP( 320, mlp_dims, dropout) # text + emo: sum, mean
        if (self.fusion_source == 5):
            if fusion_type == 'concat':
                self.classifier = MLP( 960, mlp_dims, dropout) # text + emo + img: concat
            else:
                self.classifier = MLP( 320, mlp_dims, dropout) # text + emo + img: sum, mean
        if (self.fusion_source == 6):
            self.classifier = MLP( 640, mlp_dims, dropout) # text + emo + img: sum, mean
    def forward(self, **kwargs):
        #print(f"dataloader's key features: {kwargs.keys()}")
        inputs = kwargs['content']
        masks = kwargs['content_masks']        
        category = kwargs['category']
        imgs = kwargs['img']        
        emotion = kwargs['emotion']        
        metadata = kwargs['metadata']

        #breakpoint()
        if self.emb_type == "bert":
            init_feature = self.bert(inputs, attention_mask = masks)[0]
        elif self.emb_type == 'w2v':
            init_feature = inputs
        
        feature, _ = self.attention(init_feature, masks)
        idxs = torch.tensor([index for index in category]).view(-1, 1).cuda()
        domain_embedding = self.domain_embedder(idxs).squeeze(1)

        gate_input_feature = feature
        gate_input = torch.cat([domain_embedding, gate_input_feature], dim = -1)
        gate_value = self.gate(gate_input)

        shared_feature = 0
        for i in range(self.num_expert):
            tmp_feature = self.expert[i](init_feature)
            shared_feature += (tmp_feature * gate_value[:, i].unsqueeze(1))
        #breakpoint()
        imgs_feature = imgs
        
        if (self.fusion_source == 0):
            shared_feature = shared_feature # text
        if (self.fusion_source == 1):
            imgs_feature = self.resize_img(imgs_feature)  ## resize img -> 4096 - 2742 - 320:
            shared_feature =  self.norm_text(shared_feature)
            
            ## fusion: concat
            if self.fusion_type == 'concat':
                shared_feature = torch.cat((imgs_feature ,shared_feature ), -1) ## img + text
                
            elif self.fusion_type == 'mean':
            # ## fusion: mean
                stack_vector = torch.stack([imgs_feature,shared_feature])
                shared_feature = torch.mean(stack_vector, dim=0)
            
            # ##fusion : sum
            elif self.fusion_type == 'add':
                shared_feature = torch.add(imgs_feature ,shared_feature) ## img + text
        
        if (self.fusion_source == 2):
            shared_feature =  self.norm_text(shared_feature)
            metadata = self.resize_meta(metadata)
            imgs_feature = self.resize_img(imgs_feature)  ## resize img -> 4096 - 2742 - 320
            
            ## concat
            if self.fusion_type == 'concat':
                shared_feature = torch.cat((shared_feature,imgs_feature , metadata ), -1) ## img + text + meta
            
            ## fusion: mean
            elif self.fusion_type == 'mean':
                stack_vector = torch.stack([metadata,shared_feature,imgs_feature ])
                shared_feature = torch.mean(stack_vector, dim=0)

            # ##fusion : sum
            elif self.fusion_type == 'add':
                shared_feature = torch.add(metadata ,shared_feature) 
                shared_feature = torch.add(shared_feature, imgs_feature) 
        if (self.fusion_source == 3):   # meta+ text
            shared_feature =  self.norm_text(shared_feature)
            metadata = self.resize_meta(metadata)
            ## fusion: concat
            if self.fusion_type == 'concat':
                shared_feature = torch.cat((metadata ,shared_feature ), -1) 
            
            ## fusion: mean
            elif self.fusion_type == 'mean':
                stack_vector = torch.stack([metadata,shared_feature])
                shared_feature = torch.mean(stack_vector, dim=0)

            ##fusion : sum
            elif self.fusion_type == 'add':
                shared_feature = torch.add(metadata ,shared_feature) 
        if (self.fusion_source == 4):## emotion + text
            shared_feature =  self.norm_text(shared_feature)
            #emotion = self.resize_emo(emotion)
            # Convert the 'emotion' tensor to the data type expected by self.resize_emo
            expected_dtype = self.resize_emo[0].weight.dtype
            emotion = emotion.to(expected_dtype)
    
            # Resize the 'emotion' tensor using self.resize_emo
            emotion = self.resize_emo(emotion)
            
            ## concat
            if self.fusion_type == 'concat':
                shared_feature = torch.cat((emotion ,shared_feature ), -1)
            
            ## mean
            elif self.fusion_type == 'mean':
                stack_vector = torch.stack([emotion,shared_feature])
                shared_feature = torch.mean(stack_vector, dim=0)
            
            ## sum
            elif self.fusion_type == 'add':
                shared_feature = torch.add(emotion ,shared_feature)
        if (self.fusion_source == 5):
            shared_feature =  self.norm_text(shared_feature)
            # Convert the 'emotion' tensor to the data type expected by self.resize_emo
            expected_dtype = self.resize_emo[0].weight.dtype
            emotion = emotion.to(expected_dtype)
            # Resize the 'emotion' tensor using self.resize_emo
            emotion = self.resize_emo(emotion)
            imgs_feature = self.resize_img(imgs_feature)  ## resize img -> 320
            
            ## concat
            if self.fusion_type == 'concat':
                shared_feature = torch.cat((shared_feature,imgs_feature , emotion ), -1) ## img + text + emotion
            
            ## fusion: mean
            elif self.fusion_type == 'mean':
                # stack_vector = torch.stack([emotion,shared_feature,imgs_feature])
                # shared_feature = torch.mean(stack_vector, dim=0)
                
                stack_vector_1 = torch.stack([emotion, imgs_feature])
                mean_auxi = torch.mean(stack_vector_1, dim=0)
                
                stack_vector_2 = torch.stack([shared_feature,mean_auxi])
                shared_feature = torch.mean([stack_vector_2,shared_feature])

            # ##fusion : sum
            elif self.fusion_type == 'add':
                shared_feature = torch.add(emotion ,shared_feature) 
                shared_feature = torch.add(shared_feature, imgs_feature)
        if (self.fusion_source == 6):
            shared_feature =  self.norm_text(shared_feature)
            #emotion = self.resize_emo(emotion)
            # Convert the 'emotion' tensor to the data type expected by self.resize_emo
            expected_dtype = self.resize_emo[0].weight.dtype
            emotion = emotion.to(expected_dtype)
            # Resize the 'emotion' tensor using self.resize_emo
            emotion = self.resize_emo(emotion)
            imgs_feature = self.resize_img(imgs_feature)  ## resize img -> 320
            
            ## concat(mean(shared,emo), mean(share,img))
            stack_vector = torch.stack([emotion,shared_feature])
            shared_feature1 = torch.mean(stack_vector, dim=0)
            
            stack_vector = torch.stack([imgs_feature,shared_feature])
            shared_feature2 = torch.mean(stack_vector, dim=0)
            
            shared_feature = torch.cat((shared_feature1, shared_feature2),-1)
            
        label_pred = self.classifier(shared_feature)
        
        return torch.sigmoid(label_pred.squeeze(1))

class Trainer():
    def __init__(self,
                fusion_source , 
                fusion_type,
                 emb_dim,
                 mlp_dims,
                 bert,
                 use_cuda,
                 with_emotion,
                 cat_quantity,
                 lr,
                 dropout,
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 save_param_dir,
                 emb_type = 'bert', 
                 loss_weight = [1, 0.006, 0.009, 5e-5],
                 early_stop = 5,
                 epoches = 100, 

                 ):
        self.with_emotion = with_emotion
        self.cat_quantity = cat_quantity
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.loss_weight = loss_weight
        self.use_cuda = use_cuda

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.bert = bert
        self.dropout = dropout
        self.emb_type = emb_type
        self.fusion_source = fusion_source
        self.fusion_type = fusion_type
        
        if not os.path.exists(save_param_dir):
            self.save_param_dir = os.makedirs(save_param_dir)
        else:
            self.save_param_dir = save_param_dir
        

    def train(self, logger = None):
        if(logger):
            logger.info('start training......')
        self.model = MultiDomainFENDModel(self.emb_dim, self.mlp_dims, self.bert, self.dropout, self.emb_type, self.fusion_source, self.fusion_type)
        if self.use_cuda:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.98)
        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda, self.with_emotion)
                label = batch_data['label']
                category = batch_data['category']

                optimizer.zero_grad()
                label_pred = self.model(**batch_data)
                loss =  loss_fn(label_pred, label.float()) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if(scheduler is not None):
                    scheduler.step()
                avg_loss.add(loss.item())
            
            
            results = self.test(self.val_loader)
            print('5_domain_concat_type1 - VAL Epoch {};  VAL_AUC {}'.format(epoch + 1,  results['metric']))
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                    os.path.join(self.save_param_dir, 'parameter_mdfend.pkl'))
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_mdfend.pkl')))
        results = self.test(self.test_loader)
        print(results)
        return results, os.path.join(self.save_param_dir, 'parameter_mdfend.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        category = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda, self.with_emotion)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                batch_label_pred = self.model(**batch_data) 

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
        
        return metrics(label, pred, category, self.category_dict)
