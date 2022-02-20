

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
from IPython.display import clear_output

from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
import torchvision.models as TVmodels
from tqdm import tqdm
from time import sleep
import pickle

SEED = 1

# CUDA?
cuda = torch.cuda.is_available()

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)
    


def loadMNIST():
    #### load mnist data and return pytorch data loaders
    train = MNIST('./data', train=True, download=True, transform=transforms.Compose([
                                                                                     transforms.ToTensor(), # ToTensor does min-max normalization.
                                                                                     ]), )

    test = MNIST('./data', train=False, download=True, transform=transforms.Compose([
                                                                                     transforms.ToTensor(), # ToTensor does min-max normalization. 
                                                                                     ]), )
  

    # Create DataLoader
    dataloader_args = dict(shuffle=True, batch_size=256,num_workers=0, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    train_loader = dataloader.DataLoader(train, **dataloader_args)
    test_loader = dataloader.DataLoader(test, **dataloader_args)
    return train_loader, test_loader

def loadFashMNIST():
    #### load fashion mnist data and return pytorch data loaders
    train = FashionMNIST('./data', train=True, download=True, transform=transforms.Compose([
                                                                                     transforms.ToTensor(), # ToTensor does min-max normalization.
                                                                                     ]), )

    test =  FashionMNIST('./data', train=False, download=True, transform=transforms.Compose([
                                                                                     transforms.ToTensor(), # ToTensor does min-max normalization. 
                                                                                     ]), )
  

    # Create DataLoader
    dataloader_args = dict(shuffle=True, batch_size=256,num_workers=0, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    train_loader = dataloader.DataLoader(train, **dataloader_args)
    test_loader = dataloader.DataLoader(test, **dataloader_args)
    return train_loader, test_loader

## Pytorch NN model definitions



class model_2nd_CNN(nn.Module):
    def __init__(self):
        super(model_2nd_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x = F.avg_pool2d(x1, 2, 2)
        x2 = F.relu(self.conv2(x))
        x = F.avg_pool2d(x2, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x= F.softmax(x, dim=1)
        return x, x2, x1

    
class LeNet(nn.Module):
  def __init__(self):
      super(LeNet, self).__init__()
      self.conv1 = nn.Conv2d(1, 6, (5,5), padding=2)
      self.conv2 = nn.Conv2d(6, 16, (5,5))
      self.fc1   = nn.Linear(16*5*5, 120)
      self.fc2   = nn.Linear(120, 84)
      self.fc3   = nn.Linear(84, 10)
  def forward(self, x):
      x1 = F.relu(self.conv1(x))
      x = F.max_pool2d(x1, 2, 2)
      x2 = F.relu(self.conv2(x))
      x = F.max_pool2d(x2, 2, 2)
      x = x.view(-1, 16*5*5)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      x= F.softmax(x, dim=1)
      # x = self.fc2(x)
      return x, x2, x1



    
def trainModel(model, EPOCHS, modName,train_loader,test_loader):
    ### Train model storing the best model into a file as per modName
    if cuda:
        model.cuda() # CUDA!
    optimizer = optim.Adam(model.parameters(), lr=1e-3) 

    losses = []
    bestModel = "None"
    model.train()
    best_acc = 0
    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Get Samples
            data, target = Variable(data), Variable(target)

            if cuda:
                data, target = data.cuda(), target.cuda()

            # Init
            optimizer.zero_grad()

            # Predict
            y_pred = model(data)[0]

            # Calculate loss
            loss = F.cross_entropy(y_pred, target)
            losses.append(loss.cpu().data)
    #         losses.append(loss.cpu().data[0])        
            # Backpropagation
            loss.backward()
            optimizer.step()


            # Display
            if batch_idx % 100 == 1:
                print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1,
                    EPOCHS,
                    batch_idx * len(data), 
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), 
                    loss.cpu().data), 
                    end='')
        # Eval
        evaluate_x = Variable(test_loader.dataset.data.type_as(torch.FloatTensor()))
        evaluate_y = Variable(test_loader.dataset.targets)
        if cuda:
            evaluate_x, evaluate_y = evaluate_x.cuda(), evaluate_y.cuda()

        model.eval()
        output = model(evaluate_x[:,None,...])[0]
        pred = output.data.max(1)[1]
        d = pred.eq(evaluate_y.data).cpu()
        accuracy = d.sum().type(dtype=torch.float64)/d.size()[0]
        
        # save best
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({'epoch': epoch,
                  'model': model.state_dict(),
                  'optimizer': optimizer.state_dict()
                 }, f'{modName}fin_epoch_{epoch}.pth')
                    #  }, '{}/MNIST_epoch_{}.pth'.format(save_path, epoch))
            bestModel = f'{modName}fin_epoch_{epoch}.pth'
            
            print('\r Best model saved.\r')

        print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Test Accuracy: {:.4f}%'.format(
            epoch+1,
            EPOCHS,
            len(train_loader.dataset), 
            len(train_loader.dataset),
            100. * batch_idx / len(train_loader), 
            loss.cpu().data,
            accuracy*100,
            end=''))
    del evaluate_x
    del evaluate_y
    del y_pred
    return bestModel #model


def classificationImageSaver(model, test_loader,all_size,iters, gamma, noiseMapPath):
    ### save classification images
    
    
    # test_loader = test_loaderMNIST
    batch_size = 10000

    SNR = gamma

    stats = dict()
    for i in range(10):
        stats[i] = 0    

    avgs = torch.zeros(iters, 10, 28*28)

    for kk in range(iters):
      clear_output(wait=True)
      print((kk/iters)*100)

     
      z_orig = torch.rand(all_size,  28, 28)
      z_orig.cuda()
      signal = test_loader.dataset.data.type(torch.FloatTensor)
      signal =signal/256

      z = signal*SNR + (1-SNR)*z_orig
      z.cuda()

      all_preds = []
      all_confs = []

      for k in range(0,all_size, batch_size):
          # y_pred = model(z[k:k+batch_size].cuda())[0]
          y_pred = model(z[:,None,...].cuda())[0]

          conf, pred = y_pred.data.max(1)


          all_preds.append(pred)
          all_confs.append(conf)

      preds = torch.cat(all_preds)
      confs = torch.cat(all_confs)    

    #   weighting
      tt = confs[:,None].repeat(1,784)
      uu = z.view(-1,28*28)
      z = uu* tt.type(torch.FloatTensor)
      z = z.view(-1,1,28,28)

      for i in range(10):
        stats[i] += torch.sum(preds==i)
      
        a = torch.nanmean(z_orig[preds==i] , dim=0) - torch.nanmean(z_orig[preds!=i] , dim=0) 
        avgs[kk, i] = a.reshape(28*28)

    # import os

    # setting NANs to 0
    avgs[avgs != avgs] = 0 

    dd = torch.mean(avgs, dim=0)
    torch.save(dd, noiseMapPath) #f'LN_noise_maps_{SNR}.pt')


def averageDSImageSaver(model, test_loader, noiseMapPath):
    ### save classification images
    #version average real image (from dataset)
    batch_size = 10000
    all_size = 10000
    iters = 1 #20




    avgs = torch.zeros(iters, 10, 28*28)

    for kk in range(iters):
      clear_output(wait=True)
      print((kk/iters)*100)

      signal = test_loader.dataset.data.type(torch.FloatTensor)
      signal =signal/256

      z = signal #*SNR + (1-SNR)*z_orig
      z.cuda()

      all_preds = []
      all_confs = []

      for k in range(0,all_size, batch_size):
          # y_pred = model(z[k:k+batch_size].cuda())[0]
          y_pred = model(z[:,None,...].cuda())[0]

          conf, pred = y_pred.data.max(1)


          all_preds.append(pred)
          all_confs.append(conf)

      preds = torch.cat(all_preds)
      confs = torch.cat(all_confs)    

    #   weighting
      tt = confs[:,None].repeat(1,784)
      uu = z.view(-1,28*28)
      z = uu* tt.type(torch.FloatTensor)
      z = z.view(-1,1,28,28)

      for i in range(10):
#         stats[i] += torch.sum(preds==i)
        a = torch.nanmean(z[preds==i] , dim=0) #- torch.nanmean(z_orig[preds!=i] , dim=0) 
        avgs[kk, i] = a.reshape(28*28)

    # import os

    # setting NANs to 0
#     avgs[avgs != avgs] = 0 

    dd = torch.mean(avgs, dim=0)
    torch.save(dd, noiseMapPath)#f'junkTest.pt')
    
    
def saveClassImageGraphs(noiseTensFile,figPath,model):
    f, axarr = plt.subplots(2, 5)
  
    f.set_figheight(7)
    f.set_figwidth(18)
    plt.close()



    # plotting  
#     save_path = './graphs/'
    import os

    # setting NANs to 0
#     avgs[avgs != avgs] = 0 

    # dd = torch.mean(avgs, dim=0)
    dd = torch.load(noiseTensFile)#'test_noise_maps.pt' )
    # torch.save(dd, 'test_noise_maps.pt')
    #dd = dd - grand_mean
    for kk in range(10):

#       fig = plt.figure()
      
      a = dd[kk]

      a = (a -a.min()) / (a.max() -a.min())
      a = a.view(-1,28)


      b = model(a[None,None,...].cuda())[0]

      
      conf, c = b.data.max(1) #[1]

      axarr[kk//5, kk%5].set_title(f'Class: {str(kk)}  -  pred: {str(c.cpu().data[0].numpy())}') #- |pred|: {stats[kk]}') 
    
      axarr[kk//5, kk%5].imshow(a) #, cmap = 'gray')
        
      # fig.savefig(os.path.join(save_path, str(kk)+'-.png'))
    
    f.savefig(figPath)#f'{save_path}1MAllNoise.png')  

    
def plot_confusion_matrix( y_pred, y_true, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
  
    import sklearn
    from sklearn import metrics
#     cf = sklearn.metrics.confusion_matrix(pred.numpy(), gt.numpy())  
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
  
    # Compute confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_pred, y_true)

    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] )
#         print("Normalized confusion matrix")
    else:
        pass

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.set(xticks=np.arange(0,10),#-0.5,
           yticks=np.arange(0,10),#cm.shape[0]+1)-0.5,
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Predicted label',
           xlabel='True label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()


def displayDDClassifier(storedClassImages,test_loader,average_name = "white noise"):
    noise_avg = torch.load(storedClassImages)

    avg_vec = np.stack(noise_avg).reshape(10, -1)

    print(avg_vec.shape)

    pred = []
    gt = []
    templates = torch.from_numpy(avg_vec).permute(1, 0)
    templates = templates / torch.norm(templates, p=2, dim=0)

    for batch_idx, (data, target) in enumerate(test_loader):

        x = data.view(data.shape[0], -1)
        x = x / torch.cat([torch.norm(x, p=2, dim=1).view(-1,1)] * x.shape[1], 1)
        pred.append(torch.mm(x, templates).max(1)[1])
        gt.append(target)
    pred = torch.cat(pred)
    gt = torch.cat(gt)

    acc = (pred==gt).sum().type(dtype=torch.float64) / len(gt)
    print('Accuracy: ', acc.numpy())
    return plot_confusion_matrix(pred.numpy(), gt.numpy(), classes=list(range(10)), normalize=True,  title=f'Normalized confusion matrix using {average_name} average')
# displayDDClassifier('2nd_0.3.pt',test_loaderMNIST)


# get white-noise activation for layer activation
def GetWNActivation(model,FileName,parts = 20):
    for fileNum in range(1,parts+1):
        batch_size = 250
        all_size = 50000#100000
        iters = 1 #100
        num_cls = 10
        stats = {} # recording the bias
        fc_act_noise = {}
        conv3_act_noise = {}
        conv2_act_noise = {}
        conv1_act_noise = {}
        noise = {}
        clear_output(wait=True)
        print(f'File {fileNum}/{parts}' )
        for i in range(num_cls):
            stats[i] = 0
            fc_act_noise[i] = []
            conv3_act_noise[i] = []
            conv2_act_noise[i] = []
            conv1_act_noise[i] = []
            noise[i] = []

        for kk in range(iters):
#             print(kk)
            z = torch.rand(all_size, 1, 28, 28)
       
            with tqdm(total=all_size//batch_size, file=sys.stdout) as pbar:
                for k in range(0, all_size, batch_size):
                    with torch.no_grad():
                        cur_data = z[k:k+batch_size]
                        if cuda:
                            cur_data = cur_data.cuda()
       
                        out, conv2_out, conv1_out = model(cur_data)
  
                    pred = out.max(1)[1]
                    for i in range(num_cls):
        
                        conv2_act_noise[i].append(conv2_out[pred == i].cpu())
                        conv1_act_noise[i].append(conv1_out[pred == i].cpu())
                        noise[i].append(cur_data[pred == i].cpu())
                        stats[i] += (pred == i).sum()
#                     print(f'0st class count {len(conv2_out[pred == 0])}')
                    pbar.update(1)


        
        for i in range(num_cls):
     
            conv2_act_noise[i] = torch.cat(conv2_act_noise[i]).nanmean(0)
            conv1_act_noise[i] = torch.cat(conv1_act_noise[i]).nanmean(0)
            noise[i] = torch.cat(noise[i]).nanmean(0)

        # save noise activation results
        noise_acts = {}
        # noise_acts['fc'] = fc_act_noise
        # noise_acts['conv3'] = conv3_act_noise
        noise_acts['conv2'] = conv2_act_noise
        noise_acts['conv1'] = conv1_act_noise
        noise_acts['img'] = noise
        noise_acts['stats'] = stats

        # with open('noise_acts10.pkl', 'wb') as f: 
        with open(f'{FileName}{fileNum}.pkl', 'wb') as f: 
            pickle.dump(noise_acts, f)
# GetWNActivation(modelLN,"noise_acts")




def LoadNoiseAndPlotActs(FileName,parts = 20):
    # loading noise act from multiple files
#     file_num = 10
    file_num = parts
    fc_act_noise = {}
 
    num_cls=10
    conv2_act_noise = {}
    conv1_act_noise = {}
    noise = {}

    for i in range(num_cls):
  
        conv2_act_noise[i] = []
        conv1_act_noise[i] = []
        noise[i] = []

    for i in range(1, file_num+1):
        with open(f'{FileName}{i}.pkl', 'rb') as f:
            cur_data = pickle.load(f)
   
        conv2_act = cur_data['conv2']
        conv1_act = cur_data['conv1']
        noise_img = cur_data['img']
        for k in range(num_cls):
  
            conv2_act_noise[k].append(conv2_act[k][None, ...])
            conv1_act_noise[k].append(conv1_act[k][None, ...])
            noise[k].append(noise_img[k][None, ...])

    for i in range(num_cls):
    #     fc_act_noise[i] = torch.cat(fc_act_noise[i]).mean(0)
    #     conv3_act_noise[i] = torch.cat(conv3_act_noise[i]).mean(0)
        conv2_act_noise[i] = torch.cat(conv2_act_noise[i]).nanmean(0)
        conv1_act_noise[i] = torch.cat(conv1_act_noise[i]).nanmean(0)
        noise[i] = torch.cat(noise[i]).mean(0)
    print("conv2 layer")
    figure(num=None, figsize=(12, 5), dpi=80, facecolor='w', edgecolor='k')
    for i in range(num_cls):
        plt.subplot(2, 5, i+1)
        plt.title(f'class {i}')
        plt.imshow(conv2_act_noise[i].nanmean(0).numpy())
    plt.show()
    print("conv1 layer")
    figure(num=None, figsize=(12, 5), dpi=80, facecolor='w', edgecolor='k')
    for i in range(num_cls):
        plt.subplot(2, 5, i+1)
        plt.title(f'class {i}')
        plt.imshow(conv1_act_noise[i].nanmean(0).numpy())
    plt.show()
# LoadNoiseAndPlotActs("noise_acts")


def LoadNoiseAndreturn(FileName,parts = 20):
    # loading noise act from multiple files
#     file_num = 10
    file_num = parts
    fc_act_noise = {}
    # conv3_act_noise = {}
    num_cls=10
    conv2_act_noise = {}
    conv1_act_noise = {}
    noise = {}

    for i in range(num_cls):
    #     fc_act_noise[i] = []
    #     conv3_act_noise[i] = []
        conv2_act_noise[i] = []
        conv1_act_noise[i] = []
        noise[i] = []

    for i in range(1, file_num+1):
        with open(f'{FileName}{i}.pkl', 'rb') as f:
            cur_data = pickle.load(f)
    
        conv2_act = cur_data['conv2']
        conv1_act = cur_data['conv1']
        noise_img = cur_data['img']
        for k in range(num_cls):
  
            conv2_act_noise[k].append(conv2_act[k][None, ...])
            conv1_act_noise[k].append(conv1_act[k][None, ...])
            noise[k].append(noise_img[k][None, ...])

    for i in range(num_cls):
    
        conv2_act_noise[i] = torch.cat(conv2_act_noise[i]).nanmean(0).nanmean(0)
        conv1_act_noise[i] = torch.cat(conv1_act_noise[i]).nanmean(0).nanmean(0)
        noise[i] = torch.cat(noise[i]).mean(0).nanmean(0)
    return conv2_act_noise, conv1_act_noise,   noise



# check model activation pattern on real images (training data)
def GetRealActivation(model,FileName,train_loader ):
    # actsFileName = 'train_data_acts'
    conv2_act_gt = {}
    conv1_act_gt = {}
    num_cls=10
    # pred based
   
    conv2_act_pred = {}
    conv1_act_pred = {}
#     train_loader = train_loaderMNIST

    for i in range(num_cls):

        conv2_act_gt[i] = []
        conv2_act_pred[i] = []
        conv1_act_gt[i] = []
        conv1_act_pred[i] = []

    with torch.no_grad():
        with tqdm(len(train_loader), file=sys.stdout) as pbar:
            for batch_idx, (data, target) in enumerate(train_loader):
                if cuda:
                    data, target = data.cuda(), target.cuda()
                out,   conv2_out, conv1_out = model(data)
                pred = out.max(1)[1]

                for i in range(num_cls):
    
                    conv2_act_gt[i].append(conv2_out[target == i].cpu())
                    conv2_act_pred[i].append(conv2_out[pred == i].cpu())
                    conv1_act_gt[i].append(conv1_out[target == i].cpu())
                    conv1_act_pred[i].append(conv1_out[pred == i].cpu())
                pbar.update(1)

    for i in range(num_cls):

        conv2_act_gt[i] = torch.cat(conv2_act_gt[i]).mean(0)
        conv1_act_gt[i] = torch.cat(conv1_act_gt[i]).mean(0)
    #     conv3_act_pred[i] = torch.cat(conv3_act_pred[i]).mean(0)
        conv2_act_pred[i] = torch.cat(conv2_act_pred[i]).mean(0)
        conv1_act_pred[i] = torch.cat(conv1_act_pred[i]).mean(0)

    # save training data activation results
    train_data_acts = {}
 
    train_data_acts['conv2_gt'] = conv2_act_gt
    train_data_acts['conv1_gt'] = conv1_act_gt
    # train_data_acts['fc_pred'] = fc_act_pred
    # train_data_acts['conv3_pred'] = conv3_act_pred
    train_data_acts['conv2_pred'] = conv2_act_pred
    train_data_acts['conv1_pred'] = conv1_act_pred

    with open(f'{FileName}.pkl', 'wb') as f: 
        pickle.dump(train_data_acts, f)
        
# GetRealActivation(modelLN,"realDat_acts",train_loaderMNIST)

# load activation on training data
def LoadRealDatAndPlotActs(FileName,):
    num_cls=10
    with open(f'{FileName}.pkl', 'rb') as f:
        train_data_acts = pickle.load(f)
    
    conv2_act_gt = train_data_acts['conv2_gt']
    conv1_act_gt = train_data_acts['conv1_gt']
    
    conv2_act_pred = train_data_acts['conv2_pred']
    conv1_act_pred = train_data_acts['conv1_pred']
    train_data_acts.keys()


    figure(num=None, figsize=(10.5, 3), dpi=100, facecolor='w', edgecolor='k')

    print("conv2")
    figure(num=None, figsize=(12, 5), dpi=80, facecolor='w', edgecolor='k')
    for i in range(num_cls):
        plt.subplot(2, 5, i+1)
        plt.title(f'{i}')
        plt.imshow(conv2_act_gt[i].mean(0).numpy())
    plt.show()
    print("conv1")
    figure(num=None, figsize=(12, 5), dpi=80, facecolor='w', edgecolor='k')
    for i in range(num_cls):
        plt.subplot(2, 5, i+1)
        plt.title(f'{i}')
        plt.imshow(conv1_act_gt[i].mean(0).numpy())
    plt.show()
# LoadRealDatAndPlotActs("realDat_acts",)


# plot a before and after confusion matrix and accuracy when a dataset is altered by addition of white noise
def bestModelConf(model, test_loader,Tx=False,title = 'Confusion Matrix',TxData = None):
    evaluate_x = Variable(test_loader.dataset.data.type_as(torch.FloatTensor()))
    evaluate_y = Variable(test_loader.dataset.targets)
    if cuda:
        evaluate_x, evaluate_y = evaluate_x.cuda(), evaluate_y.cuda()

    model.eval()
    output = model(evaluate_x[:,None,...])[0]
    pred = output.data.max(1)[1]
#     d = pred.eq(evaluate_y.data).cpu()

    evaluate_x = Variable(test_loader.dataset.data.type_as(torch.FloatTensor())).cuda()
    output = model(evaluate_x[:,None,...])[0]
    pred = output.data.max(1)[1]
    gt = test_loader.dataset.targets
    
    
    if not Tx:  
        plot_confusion_matrix(pred.cpu().numpy(), gt.numpy(), classes=list(range(10)), normalize=True,  title=f'{title}') 
    else:

        plot_confusion_matrix(pred.cpu().numpy(), gt.numpy(), classes=list(range(10)), normalize=True,  title=f'{title} Before') 
    

        d = pred.eq(evaluate_y.data).cpu()
        beforeAcc = d.sum().type(dtype=torch.float64)/d.size()[0]
        plt.show()
       


        model.eval()


        evaluate_x = Variable(TxData)
        if cuda:
            evaluate_x, evaluate_y = evaluate_x.cuda(), evaluate_y.cuda()
        output = model(evaluate_x[:,None,...])[0]
        pred = output.data.max(1)[1]
        gt = test_loader.dataset.targets
        d = pred.eq(evaluate_y.data).cpu()
        afterAcc = d.sum().type(dtype=torch.float64)/d.size()[0]
        plot_confusion_matrix(pred.cpu().numpy(), gt.numpy(), classes=list(range(10)), normalize=True,  title=f'{title} After') 
        plt.show()
        return beforeAcc, afterAcc
    
    
# bestModelConf(modelLN, test_loaderMNIST)