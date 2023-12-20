import torch
from torch.utils import data
import uproot4 as uproot
import numpy as np
import my_tools_PN
__all__ = ['load']


def load(filename=['Hbb.root','Hcc.root'], num_data=300_000, cache_dir=r"/hpcfs/cepc/higgsgpu/wuzuofei/HiggsHadron-PFNs-gpu/root/sample_train"):

    dataset_feature=[]
    dataset_label=[]
    ic=0
    stepsize=1_000
    for fn in filename:
        num=0
        events = uproot.open(cache_dir+"/"+fn+":t1")
        #print('{:*^80}'.format(f'loading file {fn}'))
        for array in events.iterate(step_size=stepsize, entry_stop=num_data):
            event_number=len(array['pfc_truth_PID'])
            num+=event_number
            pfc_PID     =   np.array(array['pfc_truth_PID']).reshape(stepsize,1,-1)
            pfc_E       =   np.array(array['pfc_E']).reshape(stepsize,1,-1)
            #pfc_P       =   np.array(array['pfc_P']).reshape(stepsize,1,-1)
            pfc_D0      =   np.array(array['pfc_D0']).reshape(stepsize,1,-1)
            pfc_DZ      =   np.array(array['pfc_DZ']).reshape(stepsize,1,-1)
            pfc_CosTheta=   np.array(array['pfc_CosTheta']).reshape(stepsize,1,-1)
            pfc_Phi     =   np.array(array['pfc_Phi']).reshape(stepsize,1,-1)
            
            #feature[num_of_event, feature, particle]
            dataset_feature.append(np.concatenate((pfc_PID, pfc_E, pfc_D0, pfc_DZ, pfc_CosTheta, pfc_Phi), axis=1))
            num_point_feature=2 #please make sure points feature is be set in last column, and change the num when you modify the point feature
            dataset_label.append(np.ones(event_number)*ic)
            if num%10000==0:
                print('{:*^80}'.format(f'load tuples successfully ----->{ic}, {fn}, {num}'))
        ic+=1

    dataset_feature=np.concatenate(dataset_feature,axis=0)
    dataset_label=np.concatenate(dataset_label,axis=0)
    
    #just shuffle
    idx=np.random.permutation(len(dataset_label))
    dataset_feature=dataset_feature[idx]
    dataset_label=dataset_label[idx]
    
    #optional, remap pid to a float number
    my_tools_PN.remap_pids(dataset_feature,pid_i=0,error_on_unknown=False)

    dataset_feature=torch.from_numpy(dataset_feature)
    dataset_label=torch.from_numpy(dataset_label)
    dataset_feature=dataset_feature.type(torch.float32)#  Essential !!!
    dataset_label=dataset_label.type(torch.long)   #  Essential !!!
    dataset=data.TensorDataset(dataset_feature,dataset_label)
    return dataset,num_point_feature





if __name__=='__main__':
    dataset=load()
    print(dataset)


    







