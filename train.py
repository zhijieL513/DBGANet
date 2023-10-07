import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn as nn
from Mesh_Dataset import *
from model import *
from losses_and_metrics_for_mesh import *
import utils
import pandas as pd
import time
import random

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] ="0,1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def cross_entropy(logits, y):
    s = torch.exp(logits)
    logits = s / torch.sum(s, dim=1, keepdim=True)
    c = -(y * torch.log(logits)).sum(dim=-1)
    return torch.mean(c)

 
start = time.time()
if __name__ == '__main__':
 
    setup_seed(1111)
    
    ncells = 10000
    tooth_type = 'upper'
    kfold_nums = 5
    use_fold = 1
    
     train_list_path = './dataset/KFold_{}_{}_{}/train_list_{}.csv'.format(kfold_nums, tooth_type, ncells, use_fold)
     test_list_path = './dataset/KFold_{}_{}_{}/test_list_{}.csv'.format(kfold_nums, tooth_type, ncells, use_fold)
    
     labels_dir = './dataset/tooth_label'

    
    model_path = './model/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)    
    
    segnet_name = 'DBGANet'
    num_classes = 15
    num_neighbor = 32 # k的值
    num_epochs = 100
    num_workers = 32
    train_batch_size = 6
    val_batch_size = 1
    num_batches_to_print = 150
    lr = 0.001
    
    save_model_threshold = 0.9

    
    
    start = time.time()
    model_name_path = '{}_{}classes_{}k_{}epochs_{}ncells_{}batchsize_{}kfold{}_{}lr'
    model_name_path = model_name_path.format(segnet_name, num_classes, num_neighbor, num_epochs, ncells, train_batch_size, kfold_nums, use_fold, lr)
    
    model_path = os.path.join(model_path, model_name_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)  
    
    checkpoint_name = 'latest_checkpoint.tar'
    train_loss_csv_file = 'losses_{}_{}classes_{}k_{}epochs_{}ncells_{}batchsize_{}kfold{}_{}lr.csv'
    train_loss_csv_file = train_loss_csv_file.format(segnet_name, num_classes, num_neighbor, num_epochs, ncells, train_batch_size, kfold_nums, use_fold, lr)
     
    # set dataset
    training_dataset = Mesh_Dataset(data_list_path=train_list_path, labels_dir=labels_dir, num_classes=num_classes, patch_size=ncells)
    test_dataset = Mesh_Dataset(data_list_path=test_list_path, labels_dir=labels_dir, num_classes=num_classes, patch_size=ncells)
    
    train_loader = DataLoader(dataset=training_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


    model = My_Seg(num_classes=num_classes, num_neighbor=num_neighbor)
    model = torch.nn.DataParallel(model).to(device)
    
    optimizer = optim.Adam(model.parameters(), amsgrad=True, lr=lr)
 
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    losses, mdsc, msen, mppv = [], [], [], []
    test_losses, test_mdsc, test_msen, test_mppv = [], [], [], []

    best_test_dsc = 0.0

    #cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    print('Training model...')
    class_weights = torch.ones(num_classes).to(device, dtype=torch.float)
    for epoch in range(num_epochs):
        # training
        model.train()
        
        running_loss = 0.0
        running_mdsc = 0.0
        running_msen = 0.0
        running_mppv = 0.0
        loss_epoch = 0.0
        mdsc_epoch = 0.0
        msen_epoch = 0.0
        mppv_epoch = 0.0
        for i_batch, batched_sample in enumerate(train_loader):
            inputs = batched_sample['cells'].to(device, dtype=torch.float) # (B, 6, N)
            labels = batched_sample['labels'].to(device, dtype=torch.long) # (B, 1, N)
            centroids = batched_sample['barycenter'].to(device, dtype=torch.float)
            one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes) # (B, N, 15)
            labels1 = batched_sample['barycenter_label'].to(device, dtype=torch.long)
#             one_hot_labels1 = nn.functional.one_hot(labels1[:, 0, :], num_classes=num_classes)  
            
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, cents, classes, weight = model(inputs) # (B, N, 15)
 
            labels = labels.view(-1, 1)[:, 0]
            outputs1 = outputs.contiguous().view(-1, 15)

            seg_loss = F.nll_loss(outputs1, labels)
#             seg_loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
            cent_loss = torch.nn.functional.smooth_l1_loss(cents, centroids)
#             class_loss = Generalized_Dice_Loss(classes, one_hot_labels1, class_weights)
#             print(classes.shape,labels1[:, 0, :].shape)
            class_loss = cross_entropy(classes, labels1[:, 0, :])
            
            loss = seg_loss + weight[0].item()*(cent_loss + class_loss*weight[1].item())
            
            dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
            sen = weighting_SEN(outputs, one_hot_labels, class_weights)
            ppv = weighting_PPV(outputs, one_hot_labels, class_weights)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_mdsc += dsc.item()
            running_msen += sen.item()
            running_mppv += ppv.item()
            loss_epoch += loss.item()
            mdsc_epoch += dsc.item()
            msen_epoch += sen.item()
            mppv_epoch += ppv.item()
            if i_batch % num_batches_to_print == num_batches_to_print-1:  # print every N mini-batches
                print('[Epoch: {0}/{1}, Batch: {2}/{3}] loss: {4}, dsc: {5}, sen: {6}, ppv: {7}'.format(epoch+1, num_epochs, i_batch+1, len(train_loader), running_loss/num_batches_to_print, running_mdsc/num_batches_to_print, running_msen/num_batches_to_print, running_mppv/num_batches_to_print))
#                 print("train_loss:",loss.item(), cent_loss.item(), seg_loss.item())
                
                running_loss = 0.0
                running_mdsc = 0.0
                running_msen = 0.0
                running_mppv = 0.0
        # 记录每个epoch的指标
        losses.append(loss_epoch/len(train_loader))
        mdsc.append(mdsc_epoch/len(train_loader))
        msen.append(msen_epoch/len(train_loader))
        mppv.append(mppv_epoch/len(train_loader))

        # 重置指标
        loss_epoch = 0.0
        mdsc_epoch = 0.0
        msen_epoch = 0.0
        mppv_epoch = 0.0

        scheduler.step()
        
        # 测试
        model.eval()
        with torch.no_grad():
            running_test_loss = 0.0
            running_test_mdsc = 0.0
            running_test_msen = 0.0
            running_test_mppv = 0.0
            test_loss_epoch = 0.0
            test_mdsc_epoch = 0.0
            test_msen_epoch = 0.0
            test_mppv_epoch = 0.0
            for i_batch, batched_test_sample in enumerate(test_loader): 
                inputs = batched_test_sample['cells'].to(device, dtype=torch.float) # (B, 6, N)
                labels = batched_test_sample['labels'].to(device, dtype=torch.long) # (B, 1, N)
                centroids = batched_test_sample['barycenter'].to(device, dtype=torch.float)
                one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes) # (B, N, 15)
                labels1 = batched_test_sample['barycenter_label'].to(device, dtype=torch.long)
   

                outputs, cents, classes, weight = model(inputs) # (B, N, 15)
 
                labels = labels.view(-1, 1)[:, 0]
                outputs1 = outputs.contiguous().view(-1, 15)

                seg_loss = F.nll_loss(outputs1, labels)
#                 seg_loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
                cent_loss = torch.nn.functional.smooth_l1_loss(cents, centroids)
#                 class_loss = Generalized_Dice_Loss(classes, one_hot_labels1, class_weights)
                class_loss = cross_entropy(classes, labels1[:, 0, :])
    
                loss = seg_loss + weight[0].item()*(cent_loss + class_loss*weight[1].item())
                    
                dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
                sen = weighting_SEN(outputs, one_hot_labels, class_weights)
                ppv = weighting_PPV(outputs, one_hot_labels, class_weights)

                running_test_loss += loss.item()
                running_test_mdsc += dsc.item()
                running_test_msen += sen.item()
                running_test_mppv += ppv.item()
                test_loss_epoch += loss.item()
                test_mdsc_epoch += dsc.item()
                test_msen_epoch += sen.item()
                test_mppv_epoch += ppv.item()

                if i_batch % num_batches_to_print == num_batches_to_print-1:  # print every N mini-batches
                    print('[Epoch: {0}/{1}, test batch: {2}/{3}] test_loss: {4}, test_dsc: {5}, test_sen: {6}, test_ppv: {7}'.format(epoch+1, num_epochs, i_batch+1, len(test_loader), running_test_loss/num_batches_to_print, running_test_mdsc/num_batches_to_print, running_test_msen/num_batches_to_print, running_test_mppv/num_batches_to_print))
  
                    
                    running_test_loss = 0.0
                    running_test_mdsc = 0.0
                    running_test_msen = 0.0
                    running_test_mppv = 0.0

            # record losses and metrics
            test_losses.append(test_loss_epoch/len(test_loader))
            test_mdsc.append(test_mdsc_epoch/len(test_loader))
            test_msen.append(test_msen_epoch/len(test_loader))
            test_mppv.append(test_mppv_epoch/len(test_loader))

            # reset
            test_loss_epoch = 0.0
            test_mdsc_epoch = 0.0
            test_msen_epoch = 0.0
            test_mppv_epoch = 0.0
             
            # output current status
            print('*****\nEpoch: {}/{}, loss: {}, dsc: {}, sen: {}, ppv: {}\n         test_loss: {}, test_dsc: {}, test_sen: {}, test_ppv: {}\n*****'.format(epoch+1, num_epochs, losses[-1], mdsc[-1], msen[-1], mppv[-1], test_losses[-1], test_mdsc[-1], test_msen[-1], test_mppv[-1]))
#             print("test_loss:",loss.item(), cent_loss.item(), seg_loss.item())
            
        # save the checkpoint
        torch.save({'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': losses,
                    'mdsc': mdsc,
                    'msen': msen,
                    'mppv': mppv,
                    'test_losses': test_losses,
                    'test_mdsc': test_mdsc,
                    'test_msen': test_msen,
                    'test_mppv': test_mppv},
                    os.path.join(model_path, checkpoint_name))

        # save the best model
        if test_mdsc[-1] >= save_model_threshold:
            best_test_dsc = test_mdsc[-1]
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'losses': losses,
                        'mdsc': mdsc,
                        'msen': msen,
                        'mppv': mppv,
                        'test_losses': test_losses,
                        'test_mdsc': test_mdsc,
                        'test_msen': test_msen,
                        'test_mppv': test_mppv},
                        os.path.join(model_path, format(best_test_dsc, '.4f') + "_" + str(epoch) + '.tar'))

        # save all losses and metrics data
        pd_dict = {'loss': losses, 'DSC': mdsc, 'SEN': msen, 'PPV': mppv, 'test_loss': test_losses, 'test_DSC': test_mdsc, 'test_SEN': test_msen, 'test_PPV': test_mppv}
        stat = pd.DataFrame(pd_dict)
        stat.to_csv(os.path.join(model_path, train_loss_csv_file))

        end = time.time()
        running_time = end-start
        print('Time cost: %.5fs' %running_time,running_time/3600, 'h\n')
        
print(max(test_mdsc),weight[0].item(),weight[1].item()) 