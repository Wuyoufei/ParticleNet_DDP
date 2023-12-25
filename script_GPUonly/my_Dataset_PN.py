import torch
from torch.utils import data
import uproot4 as uproot
import numpy as np
import my_tools_PN
__all__ = ['load']

class my_PN_Dataset(data.Dataset):
    def __init__(self, filename, num_data, features, points_features, root=r'/hpcfs/cepc/higgsgpu/wuzuofei/HiggsHadron-PFNs-gpu/root/sample_train') -> None:
        """ParticleNet dataset class

        Args:
            filename (list): a list of root files
            num_data (int):  number of events of one file
            features (list): features
            points_features (list): coordinates features
            root (regexp, optional): _description_. Defaults to r'/hpcfs/cepc/higgsgpu/wuzuofei/HiggsHadron-PFNs-gpu/root/sample_train'.
        """
        super().__init__()
        self.points,self.features,self.labels=self.load_data(filename, num_data, features, points_features, root)

        #preprocess
        self.Acos(idx=0)#cosTheta -> Theta
        my_tools_PN.remap_pids(self.features,pid_i=0,error_on_unknown=False)#optional, remap pid to a float number

        self.points=torch.from_numpy(self.points).type(torch.float32)
        self.features=torch.from_numpy(self.features).type(torch.float32)
        self.labels=torch.from_numpy(self.labels).type(torch.long)
    
    def __getitem__(self, index):
        return self.points[index],self.features[index],self.labels[index]
    
    def __len__(self):
        return len(self.features)


    def load_data(self,filename,num_data,features,points_features,root):
        dataset_features=[]
        dataset_points_features=[]
        dataset_labels=[]
        ic=0
        stepsize=1_000
        for fn in filename:
            num=0
            events = uproot.open(root+"/"+fn+":t1")
            for array in events.iterate(step_size=stepsize, entry_stop=num_data):
                event_number=len(array[features[0]])
                num+=event_number
                step_features=[]
                step_points_features=[]
                for feature in features:
                    step_features.append(np.array(array[feature]).reshape(event_number,1,-1))
                for points_feature in points_features:
                    step_points_features.append(np.array(array[points_feature]).reshape(event_number,1,-1))

                dataset_features.append(np.concatenate(step_features, axis=1))
                dataset_points_features.append(np.concatenate(step_points_features, axis=1))
                dataset_labels.append(np.ones(event_number)*ic)

                if num%10000==0:
                    print('{:*^80}'.format(f'load tuples successfully ----->{ic}, {fn}, {num}'))

            ic+=1
        dataset_features=np.concatenate(dataset_features,axis=0)
        dataset_points_features=np.concatenate(dataset_points_features,axis=0)
        dataset_labels=np.concatenate(dataset_labels,axis=0)

        #just shuffle
        idx=np.random.permutation(len(dataset_labels))
        dataset_points_features=dataset_points_features[idx]
        dataset_features=dataset_features[idx]
        dataset_labels=dataset_labels[idx]
        return dataset_points_features, dataset_features, dataset_labels
    
    def Acos(self,idx):
        MASK=(np.abs(self.features).sum(axis=1,keepdims=False) != 0)
        self.points[:,idx,:]=np.arccos(self.points[:,idx,:])*MASK
        print('Done! cosTheta -> Theta')

if __name__=='__main__':
    print('qwe')


    







