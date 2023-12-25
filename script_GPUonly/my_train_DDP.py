#  __        __  _____  _____ 
#  \ \      / / |__  / |  ___|
#   \ \ /\ / /    / /  | |_   
#    \ V  V /    / /_  |  _|  
#     \_/\_/    /____| |_|    
#         ParticleNet             Note: (This script can only run on one or many GPU(s)! )                  
import torch
from torch.utils import data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import tqdm
from torchsummary import summary
import my_Dataset_PN
import my_ParticleNet
import my_tools_PN
import numpy as np
import sys
import os
import time
import DDP_Config
import torch.distributed as dist
import torch.multiprocessing as mp
my_tools_PN.cheems()#means start...
try:
    SUFFIX=sys.argv[1]
except:
    default_suffix=time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime())
    print(f"Seems like you didn't type a suffix, so the [tag_'localtime({default_suffix})'] has been used as the suffix") 
    SUFFIX='tag_'+default_suffix
finally:
################################################################################################################
#Please change the hyperparameters here...
    GPU_NUM=torch.cuda.device_count()
    PROCESS=['Hbb','Hcc', 'Hgg', 'Hww', 'Hzz', 'Pll', 'Pww_l', 'Pzz_l', 'Pzz_sl']
    FEATURES=['pfc_truth_PID','pfc_E','pfc_D0','pfc_DZ','pfc_CosTheta','pfc_Phi']
    POINTS_FEATURES=['pfc_CosTheta','pfc_Phi']
    NUM_DATA_EACH_CLASS=300_0
    TRAIN_VAL_TEST=[0.8,0.1,0.1]
    EPOCHS=20
    BATCHSIZE=512
    LR=0.001
    USE_FTS_BN=True
    CONV_PARAMS=[(7, (32, 32, 32)), (7, (32, 64, 64))]
    FC_PARAMS=[(128, 0.5)]
    USE_COUNTS=True
    USE_FUSION=False
    NUM_CLASSES=len(PROCESS)
#########################################################################################################
#print info
print(f'''
suffix:         {SUFFIX}
# gpu           {GPU_NUM}
classes:        {NUM_CLASSES}
features:       {FEATURES}
points_features:{POINTS_FEATURES}
# each class    {NUM_DATA_EACH_CLASS}
train_val_test: {TRAIN_VAL_TEST}
epochs:         {EPOCHS}
batchsize:      {BATCHSIZE}
lr:             {LR}
use_fts_bn:     {USE_FTS_BN}
conv_params:    {CONV_PARAMS}
fc_params:      {FC_PARAMS}
use_counts      {USE_COUNTS}
use_fusion      {USE_FUSION}
        ''')
def main(local_rank, train_set, val_set, test_set):
    DDP_Config.init_ddp(local_rank)

    #########################################################################################################    
                                                #dataloader
    points_shape_for_one_event=train_set[0][0].shape    #(coordinates, particles)
    features_shape_for_one_event=train_set[0][1].shape  #(features_dim, particles)
    feature_dim=features_shape_for_one_event[0]         #int

    train_sampler=data.DistributedSampler(train_set)
    val_sampler=data.DistributedSampler(val_set)
    
    train_set=data.DataLoader(train_set,batch_size=BATCHSIZE,sampler=train_sampler,shuffle=False)
    val_set=data.DataLoader(val_set,batch_size=BATCHSIZE,shuffle=False,sampler=val_sampler)
    test_set=data.DataLoader(test_set,batch_size=BATCHSIZE,shuffle=False)
    #########################################################################################################


    net=my_ParticleNet.ParticleNet(input_dims=feature_dim,
                                   num_classes=NUM_CLASSES,
                                   conv_params=CONV_PARAMS,
                                   fc_params=FC_PARAMS,
                                   use_fts_bn=USE_FTS_BN,
                                   use_counts=USE_COUNTS,
                                   use_fusion=USE_FUSION
                                   )
    #net.apply(my_tools_PN.initialize_pfn)    #kaiming_init by default
    loss=nn.CrossEntropyLoss(reduction='none') #none is essential！
    loss.to(local_rank)
    net.to(local_rank)

    if local_rank==0:
        writer=SummaryWriter(f'output/log_tensorboard/log_{SUFFIX}')
        print(f'TensorBoard log path is {os.getcwd()}/output/log_tensorboard/log_{SUFFIX}')
        writer.add_graph(net,input_to_model=[torch.rand(1,*points_shape_for_one_event).to(local_rank),
                                             torch.rand(1,*features_shape_for_one_event).to(local_rank)])
        
        print(summary(net,input_size=[points_shape_for_one_event,features_shape_for_one_event]))#list of tuple when input is multi-dim

    net=nn.parallel.DistributedDataParallel(net,device_ids=[local_rank])

    optimizer=torch.optim.NAdam(net.parameters(),lr=LR)#After the  "net.to(device=device)"


    if local_rank==0:
        torch.cuda.synchronize()
        start=time.time()

    for epoch in range(EPOCHS):
        train_set.sampler.set_epoch(epoch) #Essential!!! Otherwize each GPU only get same ntuples every epoch
        #train_sampler.set_epoch(epoch)    #Same as the above, both are correct
        loss_train,acc_train=my_tools_PN.train_procedure_in_each_epoch(net,train_set,loss,optimizer,local_rank)
        loss_val,acc_val=my_tools_PN.evaluate_accuracy(net,loss,val_set,test=False)

        if local_rank==0:
            writer.add_scalars('Metric',{'loss_train':loss_train,
                                        'acc_train':acc_train,
                                        'loss_val':loss_val,
                                        'acc_val':acc_val},epoch)
            print(f'''epoch: {epoch} | acc_train: {acc_train:.3f} | acc_val: {acc_val:.3f} | loss_train: {loss_train:.3f} | loss_val: {loss_val:.3f} | 
            ''')

    if local_rank==0:
        torch.cuda.synchronize()
        end=time.time()
        print('Total time in training is ',end-start)

    if local_rank==0:
        my_tools_PN.save_net(net,SUFFIX)
        score,true_label,loss_test,acc_test=my_tools_PN.evaluate_accuracy(net,loss,test_set,test=True)
        print('{:-^80}'.format(f'loss in test is {loss_test:.3f}, accuracy in test is {acc_test:.3f},'))
        score_label=torch.argmax(score,dim=1)
        true_label=true_label.cpu().numpy()     #1d
        score_label=score_label.cpu().numpy()   #1d 
        score=score.cpu().numpy()               #2d
        my_tools_PN.plot_confusion_matrix(SUFFIX,NUM_CLASSES,true_label,score_label,classes=PROCESS)
        my_tools_PN.plot_roc(SUFFIX,NUM_CLASSES,true_label,score,PROCESS)

    dist.destroy_process_group()
################################################################################################################################### 

if __name__=='__main__':

    fimename=[i +'.root' for i in PROCESS]
    dataset=my_Dataset_PN.my_PN_Dataset(fimename,NUM_DATA_EACH_CLASS,FEATURES,POINTS_FEATURES)
    train_set,val_set,test_set=data.random_split(dataset=dataset,lengths=TRAIN_VAL_TEST)
    
    mp.spawn(main,args=(train_set, val_set, test_set), nprocs=GPU_NUM)

    print('Program Is Over !!!')









