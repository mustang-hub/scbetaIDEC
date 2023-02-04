import math
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import scanpy as sc
import time
from ZINB import MeanAct,ZINBLoss,DispAct
import numpy as np
import h5py
from sklearn import metrics
from sklearn.cluster import KMeans
import pandas as pd
import math,os
from cluster_accuracy  import cluster_acc


class VAE(nn.Module):
    def __init__(self,beita):
        super(VAE, self).__init__()
        # encoder
        self.encode = nn.Sequential(
            # nn.Linear(13488, 8192),
            # nn.ReLU(),
            # nn.Linear(8192,4096),
            # nn.ReLU(),
            # nn.Linear(4096,2048),
            # nn.ReLU(),
            nn.Linear(13166,128),
            nn.ReLU(),
            # nn.Linear(1024,512),
            # nn.ReLU(),
            # nn.Linear(512,256),
            # nn.ReLU(),
            # nn.Linear(256,128),
            # nn.ReLU(),
            # nn.Linear(256,128),
            # nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32)
        )
        self.decode_mean = nn.Sequential(
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            # nn.Linear(128, 256),
            # nn.ReLU(),
            # nn.Linear(128,256),
            # nn.ReLU(),
            # nn.Linear(256,512),
            # nn.ReLU(),
            # nn.Linear(512,1024),
            # nn.ReLU(),
            nn.Linear(128,13166),
            # nn.ReLU(),
            # nn.Linear(2048,4096),
            # nn.ReLU(),
            # nn.Linear(4096,8192),
            # nn.ReLU(),
            # nn.Linear(8192,13488),
            MeanAct()
        )
        self.decode_disp = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            # nn.Linear(128, 256),
            # nn.ReLU(),
            # nn.Linear(128, 256),
            # nn.ReLU(),
            # nn.Linear(256,512),
            # nn.ReLU(),
            # nn.Linear(512,1024),
            # nn.ReLU(),
            nn.Linear(128, 13166),
            # nn.ReLU(),
            # nn.ReLU(),
            # nn.Linear(2048, 4096),
            # nn.ReLU(),
            # nn.Linear(4096, 8192),
            # nn.ReLU(),
            # nn.Linear(8192, 13488),
            DispAct()
        )
        self.decode_pi = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            # nn.Linear(128, 256),
            # nn.ReLU(),
            # nn.Linear(128, 256),
            # nn.ReLU(),
            # nn.Linear(256,512),
            # nn.ReLU(),
            # nn.Linear(512,1024),
            # nn.ReLU(),
            nn.Linear(128, 13166),
            # nn.ReLU(),
            # nn.ReLU(),
            # nn.Linear(2048, 4096),
            # nn.ReLU(),
            # nn.Linear(4096, 8192),
            # nn.ReLU(),
            # nn.Linear(8192, 13488),
            nn.Sigmoid()
        )
        self.decode_x_bar = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            # nn.Linear(128, 256),
            # nn.ReLU(),
            # nn.Linear(128, 256),
            # nn.ReLU(),
            # nn.Linear(256,512),
            # nn.ReLU(),
            # nn.Linear(512,1024),
            # nn.ReLU(),
            nn.Linear(128, 13166),
            # nn.ReLU(),
            # nn.Linear(2048, 4096),
            # nn.ReLU(),
            # nn.Linear(4096, 8192),
            # nn.ReLU(),
            # nn.Linear(8192, 13488),
            nn.ReLU()
        )
        self.zinb_loss = ZINBLoss().cuda()
        self.beita = beita
    def forward(self,x):
        mu = self.encode(x)
        logvar = self.encode(x)
        z_ = self.reparametrize(mu,logvar)
        x_bar = self.decode_x_bar(z_)
        h_dec_mean = self.decode_mean(z_)
        h_dec_disp = self.decode_disp(z_)
        h_dec_pi = self.decode_pi(z_)
        return z_,x_bar,mu,logvar,h_dec_mean,h_dec_disp,h_dec_pi

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def loss_function(self,recon_x,x,mu,logvar):
        reconstruction_function = nn.MSELoss(size_average=False)
        BCE = reconstruction_function(recon_x, x) # mse loss
        # BCE = torch.sqrt(BCE)
        # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        return self.beita *  KLD +  BCE


class scVIDEC(nn.Module):
    def __init__(self,autoencoder = None,n_clusters= None,hidden = None,alpha = None,gamma1 = None,gamma2 = None,gamma3 = None):
        super(scVIDEC, self).__init__()
        self.autoencoder = autoencoder
        self.alpha = alpha
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.n_clusters = n_clusters
        self.mu = Parameter(torch.Tensor(n_clusters, hidden))


    def save_model(self,path):
        torch.save(self.state_dict(),path)

    def load_model_vae(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.autoencoder.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict['vae_state_dict'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.autoencoder.load_state_dict(model_dict)

    def load_model_zinb(self,path):
        pretrained_dict = torch.load(path,map_location=lambda storage,loc:storage)
        model_dict = self.autoencoder.state_dict()
        pretrained_dict = {k:v for k,v in pretrained_dict['ae_state_dict'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.autoencoder.load_state_dict(model_dict)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def save_check_point2(self, state, index, filename):
        newfile_name = os.path.join(filename,'FTcheckpoint%d.pth.tar' % index)
        torch.save(state,newfile_name)

    # 真实分布
    def forward(self,z_ij):
        """soft assignment using t-distribution q_{ij}"""
        q = 1.0 / (1.0 + torch.sum((z_ij.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q_ij = (q.t() / torch.sum(q, dim=1)).t()
        return q_ij

    # 目标分布
    def target_distribution(self,q_ij):
        """计算目标分布p_ij"""
        weight = (q_ij ** 2) / torch.sum(q_ij,0)
        p_ij =  (weight.t() / torch.sum(weight, 1)).t()
        return p_ij

    # 预训练
    def pretrain_VAE(self, x=None, X_raw=None, size_factor=None, batch_size=None, lr=None, epochs=None,
                     ae_save=True,
                     ae_weights='AE_weights.pth'):
        print(self.autoencoder)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.autoencoder.parameters()), lr=lr, amsgrad=True)
        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(X_raw), torch.Tensor(size_factor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.autoencoder.train()
        for epoch in range(epochs):
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                start_time = time.time()
                x_tensor = Variable(x_batch)
                optimizer.zero_grad()
                z_, rec_batch, mu, logvar, mean_tensor, disp_tensor, pi_tensor = self.autoencoder.forward(x_tensor)
                loss_vae = self.autoencoder.loss_function(recon_x=rec_batch,x=x_tensor,mu=mu, logvar=logvar)
                loss_vae.backward()
                optimizer.step()
                # pretrain_loss += loss_vae.item()
                end_time = time.time()
                print('Pretrain epoch [{}/{}], preVAE_loss:{:.4f},time:{}s'.format(batch_idx + 1, epoch + 1,
                                                                                   loss_vae.item()/len(x_tensor),(end_time - start_time)))
        if ae_save:
            torch.save({'vae_state_dict': self.autoencoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, ae_weights)

    def pretrain_ZINB(self, x=None, X_raw=None, size_factor=None, batch_size=None, lr=None, epochs=None, ae_save=True,
                             zinb_weights='ZINB_weights.pth'):

        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(X_raw), torch.Tensor(size_factor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(self.autoencoder)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.autoencoder.parameters()), lr=lr, amsgrad=True)

        for epoch in range(epochs):
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                start_time_ = time.time()
                x_tensor = Variable(x_batch)
                x_raw_tensor = Variable(x_raw_batch)
                sf_tensor = Variable(sf_batch)
                # z_,x_bar,mu,logvar,h_dec_mean,h_dec_disp,h_dec_pi
                _,_,_,_, mean_tensor, disp_tensor, pi_tensor = self.autoencoder.forward(x_tensor)
                loss = self.autoencoder.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor,
                                      scale_factor=sf_tensor)
                end_time_ = time.time()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('Pretrain epoch [{}/{}], ZINB loss:{:.4f},time:{}s'.format(batch_idx + 1, epoch + 1, loss.item(),(end_time_-start_time_)))

        if ae_save:
            torch.save({'ae_state_dict': self.autoencoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, zinb_weights)


    # 聚类模型scVAE_ZINB_DEC
    def fit_scVIDEC(self,X=None, X_raw=None, size_factor=None,batch_size=None,num_epochs = None,
                      lr = None,cell_lable = None,tol=None,update_interval = 1.0,save=None,scvzd="scvzed_param.pth"):
        y_ture = cell_lable
        print("Clustering stage")
        nmi_cure = []
        X = torch.tensor(X)
        X_raw = torch.tensor(X_raw)
        sf = torch.tensor(size_factor)
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad,self.parameters()),lr=lr,rho=0.95)
        # ===========================Cluster_Inital_With_Kmeans=================================
        print("Initializing cluster centers with kmeans")

        kmeans = KMeans(n_clusters=self.n_clusters)
        data = self.autoencoder(X)
        data = data[0].detach().numpy()
        y_pred = kmeans.fit_predict(data) + 1
        y_pred = y_pred.astype(float)


        y_pred_last = y_pred
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))

        if y_ture is not None:
            nmi = metrics.normalized_mutual_info_score(y_ture,y_pred)
            ari = metrics.adjusted_rand_score(y_ture,y_pred)
            acc = cluster_acc(y_ture.astype(np.int), y_pred.astype(np.int))
            print('Initializing k-means:NMI = %.4f,ARI = %.4f,ACC = %.4f' % (nmi,ari,acc))

        self.train()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))

        Final_NMI,Final_ARI,Final_epoch,Final_ACC = 0,0,0,0

        for epoch in range(num_epochs):

            if epoch % update_interval == 0:
                # 更新目标分布p
                latent_z = self.autoencoder(X)[0]
                q_ij = self.forward(latent_z)
                p_ij = self.target_distribution(q_ij).data

                # 评估聚类性能
                y_pred = torch.argmax(q_ij,dim=1).data.cpu().numpy()

                if y_ture is not None:
                    Final_NMI = nmi = metrics.normalized_mutual_info_score(y_ture,y_pred)
                    Final_ARI = ari = metrics.adjusted_rand_score(y_ture,y_pred)
                    Final_ACC = acc = cluster_acc(y_ture.astype(np.int),y_pred.astype(np.int))
                    print('Clustering   %d: NMI= %.4f, ARI= %.4f,ACC= %.4f' % (epoch + 1, nmi, ari,acc))
                    nmi_cure.append([epoch, nmi,ari,acc])

                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / num
                print('delta_label',delta_label)
                # save current model
                if (epoch > 0 and delta_label < tol ) or epoch % 10 == 0:
                    torch.save({'scvidec_dict':self.state_dict(),
                    'optimizer_stata_dict':optimizer.state_dict()}, scvzd)

                # 检查停止标准
                y_pred_last = y_pred

                if epoch > 0 and delta_label < tol:
                    print('delta_label',delta_label,'< tol',tol)
                    print("Rech tolerance threshold.Stopping training.")
                    break

            # train 1 epoch for clustering loss

            train_loss = 0.0
            loss_vae_val = 0.0
            loss_zinb_val = 0.0
            loss_cluster_val = 0.0

            for batch_idx in range(num_batch):

                xbatch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                xrawbatch = X_raw[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                sfbatch = sf[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                pbatch = p_ij[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]

                optimizer.zero_grad()
                rawinputs = Variable(xrawbatch)

                inputs = Variable(xbatch)
                sfinputs = Variable(sfbatch)
                target = Variable(pbatch)

                z_ij, x_bar, mu, logvar, h_dec_mean, h_dec_disp, h_dec_pi = self.autoencoder(inputs)

                output = self.forward(z_ij)  # 计算q_ij

                loss_vae = self.autoencoder.loss_function(recon_x=x_bar,x=inputs,mu=mu, logvar=logvar)
                # loss_vae = loss_vae * 0.2
                loss_zinb = self.autoencoder.zinb_loss(x=rawinputs,mean = h_dec_mean,disp = h_dec_disp,
                                                  pi = h_dec_pi,scale_factor = sfinputs)

                loss_function_clutser = nn.KLDivLoss(size_average=False)
                loss_cluster =loss_function_clutser(output.log(), target)  # 损失函数
                # loss_cluster = 0.3 * loss_cluster

                loss =  loss_vae  +  0.6 * loss_cluster +  loss_zinb

                loss.backward()
                optimizer.step()

                loss_cluster_val +=  loss_cluster.data
                loss_vae_val += loss_vae.data
                loss_zinb_val += loss_zinb.data

                train_loss = loss_cluster_val + loss_vae_val + loss_zinb_val

                # print('loss_vae,loss_zinb,loss_cluster',loss_vae,loss_zinb,loss_cluster)

            print("#Epoch %3d: Total: %.4f Clustering Loss: %.4f VAE Loss: %.4f ZINB Loss: %.4f" % (
                epoch + 1, train_loss/num, loss_cluster_val/num , loss_vae_val/num, loss_zinb_val/num))

        if save:
            torch.save({'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, scvzd)

        np.savetxt('/Users/mustang/PycharmProjects/pythonProject1/10XPBMC_large/10XPBMC_y_pred.txt',y_pred_last,delimiter=",")
        df1 = pd.DataFrame(nmi_cure, columns=['epochs', 'NMI','ARI','ACC'])
        df1.to_csv('/Users/mustang/PycharmProjects/pythonProject1/10XPBMC_large/10XPBMC_large_evalution_index.csv')
        return y_pred_last,Final_NMI,Final_ARI,Final_epoch,Final_ACC



# if __name__ == '__main__':
#
#     import argparse
#     parser = argparse.ArgumentParser(description='train',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--batch_size',default=256,type=int)
#     parser.add_argument('--train_epochs',default=3,type=int)
#     parser.add_argument('--save_dir',default='saves')
#     parser.add_argument('--learn_rate',default=1e-4,type=float)
#     parser.add_argument('--n_clusters',default=10,type=int)
#     parser.add_argument('--hidden',default=64,type=int)
#     parser.add_argument('--num_epochs',default=3,type=int)
#     parser.add_argument('--data_file',default='/Users/mustang/PycharmProjects/PyTorch_project/scDCC-master/data/Small_Datasets/worm_neuron_cell.h5')
#     parser.add_argument('--label_cells', default=0.1, type=float)
#     parser.add_argument('--ae_weight_file', default='AE_weights_p0_1.pth.tar')
#     parser.add_argument('--zinb_weight_file', default='ZINB_weights_p0_2.pth.tar')
#     parser.add_argument('--pretrain_epochs', default=5, type=int)
#     parser.add_argument('--scvidec_weight_file',default='scvidec_weights_p0_3.pth.tar')
#     parser.add_argument('--train_num_epochs',default=3,type=int)
#     parser.add_argument('--alpha',default=0.001,type=float)
#     parser.add_argument('--beita', default=0.1, type=float)
#     parser.add_argument('--gamma', default=0.1, type=float)
#     parser.add_argument('--update_interval',default=1.0,type=float)
#     parser.add_argument('--maxiter',default=3,type=int)
#     parser.add_argument('--tol',default=1e-3,type=float)
#     args = parser.parse_args()
#     print(args)
#     batch_size = args.batch_size
#
#     data_mat = h5py.File(args.data_file)
#     x = np.array(data_mat['X'])
#     y = np.array(data_mat['Y'])
#     data_mat.close()
#     adata = sc.AnnData(x)
#     adata.obs['Group'] = y
#
#     adata = read_dataset(adata, transpose=False, test_split=False, copy=True)
#     adata = normalize(adata, size_factors=True, normalize_input=True, logtrans_input=True)
#     input_size = adata.n_vars
#     print(adata.X.shape)
#     print(y.shape)
#     autoencoder = VAE().to(device)
#     ClusterLayer = ClusterLayer(autoencoder=autoencoder,n_clusters=args.n_clusters,hidden=args.hidden,alpha=1.0).to(device)
#
#     model = scVIDEC(autoencoder = autoencoder,clusterlayer=ClusterLayer,alpha=args.alpha,beita=args.beita,gamma=args.gamma)
#
#     scVIDEC.pretrain_VAE(self=model,x=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors,batch_size=args.batch_size, lr=args.learn_rate,epochs=args.pretrain_epochs,
#                                ae_weights=args.ae_weight_file)
#     scVIDEC.pretrain_ZINB(self=model, x=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors,
#                          batch_size=args.batch_size, lr=args.learn_rate, epochs=args.pretrain_epochs,
#                          zinb_weights=args.zinb_weight_file)
#
#     # scVIDEC.fit_scVIDEC(self=model,X=adata.X,X_raw=adata.raw.X,size_factor=adata.obs.size_factors,batch_size=args.batch_size,num_epochs = args.maxiter,
#     #             update_interval=args.update_interval,lr = 1.0,cell_lable = y,tol = args.tol,save_dir=args.scvidec_weight_file)
