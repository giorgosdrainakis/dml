import math
import random
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
import torch.utils.data as torch_data
import torch.nn.functional as F
import Tools
import Global
from torch.utils.data import random_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

def get_list_sum(mylist):
    mysum=0
    for el in mylist:
        mysum=mysum+el
    return mysum

class Subdataset:
    def __init__(self):
        self.lst=[]
        self.is_used=False
        self.id=None
        self.is_selected=False
    def get_total_classes(self):
        cl=[]
        print('Checking subset='+str(self.id))
        print('With len lst='+str(len(self.lst)))
        for batch_idx, (data, target) in enumerate(self.lst):
            geo=target.squeeze().tolist()
            #print('Target='+str(target))
            cl.extend(geo)
        return len(set(cl))

class Subdatasets:
    def __init__(self):
        self.db=[]
    def add_new(self,new_dataset):
        max_id=-1
        for sub in self.db:
            if sub.id>max_id:
                max_id=sub.id
        new_dataset.id=max_id+1
        self.db.append(new_dataset)
    def remove_one(self,wanted_number):
        if len(self.db)==wanted_number:
            pass
        else:
            smallest_id=-1
            min_set=math.inf
            for sub in self.db:
                if len(sub.lst)<min_set:
                    smallest_id=sub.id
            saved_i=-1
            for i in range(0,len(self.db)):
                if self.db[i].id==smallest_id:
                    saved_i=i
                    break
            self.db.pop(saved_i)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class SVHN_big:
    def __init__(self,subdatasetnum,mybatch,mylearnrate,sd):
        self.learning_rate=mylearnrate
        self.subdatasetnum=subdatasetnum
        self.subdatasets=Subdatasets()
        self.batch_size=mybatch
        self.noise_repeat_factor=1
        self.noise_repeats=0
        self.sd=sd
        print('SVHN big with batch-rate= '+str(mybatch)+'-'+str(mylearnrate))

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_set=torchvision.datasets.SVHN(root=Global._ROOT + Global._DATASETS_FOLDER + Global._SVHN_PATH,
                                  split='test', transform=transform, target_transform=None, download=True)

        self.test_loader = torch.utils.data.DataLoader(test_set, self.batch_size, shuffle=True)
        self.dataset_size=self.noise_repeat_factor*Tools.get_file_size_in_mb(Global._ROOT+Global._DATASETS_FOLDER+Global._SVHN_DATASET_PATH)
        self.dataset_len=0
        self.reset_subdatasets()

    def reset_subdatasets(self):
        self.subdatasets=Subdatasets()
        current_trainset=self.create_dataset()
        mini_dataset_len=len(current_trainset) # number of total images
        print('Mindataset len='+str(mini_dataset_len))
        self.dataset_len=mini_dataset_len*self.noise_repeat_factor
        train_loader = torch.utils.data.DataLoader(current_trainset, self.batch_size, shuffle=True)

        # calculate minimum batch
        mymin=1
        print('Mymin='+str(mymin))

        # eed
        D=mini_dataset_len/self.batch_size # dataset size in batches
        print('D (batches)='+str(D))
        mu = D / self.subdatasetnum # average per client dataset
        print('mu (batches)='+str(mu))
        sigma = mu * self.sd # sigma in relation to mu
        subdataset_batch_list = [] # final dataset list

        # apply gaussian distribution to eed, with N(mu,sigma)
        new_size_list = np.random.zipf(self.sd, self.subdatasetnum)
        sum = 0
        for i in new_size_list:
            sum = sum + i
        new_size_list = [x / sum for x in new_size_list]

        for i in range(0, self.subdatasetnum):
            current_list_size = get_list_sum(subdataset_batch_list)
            new_size = round(new_size_list[i] * D)
            if current_list_size + new_size <= D:
                subdataset_batch_list.append(new_size)
            else:
                subdataset_batch_list.append(max(D - current_list_size,mymin))
        # sort dataset list
        subdataset_batch_list = sorted(subdataset_batch_list)

        # in case total dataset list!=D, uniformly add difference
        if get_list_sum(subdataset_batch_list)==D:
            print('Mydiff=0, no need for normalization')
        elif get_list_sum(subdataset_batch_list)<D:
            print('Mydiff is negative, I have added less data, need to uniformly add')
            while get_list_sum(subdataset_batch_list)<D:
                i=0
                while get_list_sum(subdataset_batch_list)<D and i<len(subdataset_batch_list):
                    subdataset_batch_list[i]=subdataset_batch_list[i]+mymin
                    i=i+1
            print('My new diff is='+str(get_list_sum(subdataset_batch_list)-D))
        elif get_list_sum(subdataset_batch_list)>D:
            print('Mydiff is positive, I have added more data, need to uniformly remove')
            while get_list_sum(subdataset_batch_list)>D:
                i=0
                while get_list_sum(subdataset_batch_list)>D and i<len(subdataset_batch_list):
                    if subdataset_batch_list[i]>mymin:
                        subdataset_batch_list[i]=subdataset_batch_list[i]-mymin
                    i=i+1
            print('My new diff is='+str(get_list_sum(subdataset_batch_list)-D))

        # normalize dataset list to not contain zero values
        for i in range(0,self.subdatasetnum):
            if subdataset_batch_list[i]<mymin:
                diff=mymin-subdataset_batch_list[i]
                subdataset_batch_list[i] = mymin
                j = 0
                found = False
                while j < self.subdatasetnum and (not found):
                    if subdataset_batch_list[-j] > 2 * mymin:
                        subdataset_batch_list[-j] = subdataset_batch_list[-j] - diff
                        found = True
                    j = j + 1

        print('Batch list='+str(subdataset_batch_list))
        mysum=0
        for el in subdataset_batch_list:
            mysum=mysum+el
        print('Batch list size='+str(mysum))
        # unsort/shuffle dataset list
        random.shuffle(subdataset_batch_list)

        # assign data based on non-eed dataset list
        unique_id=0
        subdataset=Subdataset()
        for batch_idx, (data, target) in enumerate(train_loader):
            if unique_id < self.subdatasetnum:
                if len(subdataset.lst) < subdataset_batch_list[unique_id]:
                    pass
                else:
                    self.subdatasets.add_new(subdataset)
                    unique_id=unique_id+1
                    subdataset=Subdataset()
                subdataset.lst.append(((data, target)))
            else:
                # again due to truncacion we add all remainders (few batches) to the last record
                print('Adding extra trunc '+str(len(subdataset.lst)))
                subdataset.lst.append(((data, target)))
        self.subdatasets.add_new(subdataset)

        print('Before removal='+str(len(self.subdatasets.db)))
        #remove subdatasets if more than subdataset num due to truncactions
        self.subdatasets.remove_one(self.subdatasetnum)
        print('Found a total of subdatasets='+str(len(self.subdatasets.db)))

        # debug print results
        mystr=''
        mysum=0
        for sub in self.subdatasets.db:
            mystr=mystr+str(len(sub.lst))+','
            mysum=mysum+len(sub.lst)
        print('Reset subdatasets output= '+mystr+'- with size='+str(mysum))
        print('Total='+str(mysum))

        # non-iid shuffle
        random.shuffle(self.subdatasets.db)

    def create_dataset(self):
        transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                 AddGaussianNoise(0., 0.0001)])

        train_set = torchvision.datasets.SVHN(root=Global._ROOT + Global._DATASETS_FOLDER + Global._SVHN_PATH,
                                                  split='extra', transform=transform, target_transform=None,
                                                  download=True)
        return train_set

    def is_dataset_depleted(self):
        for sub in self.subdatasets.db:
            if not sub.is_used:
                return False
        return True

    def is_dataset_depleted_with_noise(self):
        return (self.noise_repeat_factor<=self.noise_repeats)

    def unmark_used_subdataset(self,id):
        found=False
        for sub in self.subdatasets.db:
            if sub.id==id:
                found=True
                # we only care for this noise round's datasets
                if sub.is_selected:
                    sub.is_used=False
        if not found:
            print(str('debug ERROR used subdataset not found!'))

    def mark_used_subdataset(self,id):
        found=False
        for sub in self.subdatasets.db:
            if sub.id==id:
                found = True
                # we only care for this noise round's datasets
                if sub.is_selected:
                    sub.is_used=True
        if not found:
            print(str('debug ERROR used subdataset not found!'))

    def debug_subdatasets(self):
        mylist=[]
        for sub in self.subdatasets.db:
            mystr=str(sub.id)+'-'+str(sub.is_selected)+'-'+str(sub.is_used)
            mylist.append(mystr)
        return mylist

    def unmark_all_selected(self):
        for sub in self.subdatasets.db:
            sub.is_selected=False

    def mark_selected_subdataset(self,id):
        found=False
        for sub in self.subdatasets.db:
            if sub.id==id:
                sub.is_selected=True
                found=True
        if not found:
            print(str('debug ERROR used subdataset not found!'))

    def unmark_selected_subdataset(self,id):
        found=False
        for sub in self.subdatasets.db:
            if sub.id==id:
                sub.is_selected=False
                found=True
        if not found:
            print(str('debug ERROR used subdataset not found!'))

    def get_next_subset(self):
        for sub in self.subdatasets.db:
            if (not sub.is_used) and (not sub.is_selected):
                return sub
        return None

    def get_next_subset_repeat(self):
        sub_to_return=None
        for sub in self.subdatasets.db:
            if (not sub.is_used) and (not sub.is_selected):
                sub_to_return=sub
        if sub_to_return is None: #reset datasets
            self.noise_repeats = self.noise_repeats+1
            if self.is_dataset_depleted_with_noise():
                sub_to_return=None
            else:
                self.reset_subdatasets()
                sub_to_return=self.get_next_subset_repeat()
        return sub_to_return

    def get_list_of_all_datasets(self):
        mylist=[]
        for sub in self.subdatasets.db:
            mylist.append(sub.id)
        mylist=sorted(mylist)
        return mylist

    def get_list_of_used_datasets(self):
        mylist=[]
        for sub in self.subdatasets.db:
            if sub.is_used:
                mylist.append(sub.id)
        mylist=sorted(mylist)
        return mylist

    def do_test(self,model):
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        model.eval()
        test_loss = 0
        correct = 0
        actuals = []
        predictions = []
        with torch.no_grad():
            for data, target in self.test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.argmax(1, keepdim=True)
                actuals.extend(target.view_as(pred))
                predictions.extend(pred)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        torch.save(model.state_dict(), Global._ROOT+Global._MODELS_FOLDER+'SVHN.pt')
        #conf=confusion_matrix(actuals, predictions, labels=labels)
        return test_loss, float(100 * correct / len(self.test_loader.dataset))

class SVHN_big_noniid:
    def __init__(self,subdatasetnum,mybatch,mylearnrate,splits,sd):
        self.learning_rate=mylearnrate
        self.subdatasetnum=subdatasetnum
        self.subdatasets=Subdatasets()
        self.batch_size=mybatch
        self.noise_repeat_factor=1
        self.noise_repeats=0
        self.splits=splits
        self.sd=sd
        print('SVHN big with batch-rate= '+str(mybatch)+'-'+str(mylearnrate))

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_set=torchvision.datasets.SVHN(root=Global._ROOT + Global._DATASETS_FOLDER + Global._SVHN_PATH,
                                  split='test', transform=transform, target_transform=None, download=True)

        self.test_loader = torch.utils.data.DataLoader(test_set, self.batch_size, shuffle=True)
        self.dataset_size=self.noise_repeat_factor*Tools.get_file_size_in_mb(Global._ROOT+Global._DATASETS_FOLDER+Global._SVHN_DATASET_PATH)
        self.dataset_len=0
        self.reset_subdatasets()

    def reset_subdatasets(self):
        self.subdatasets=Subdatasets()
        current_trainset=self.create_dataset()
        mini_dataset_len=len(current_trainset)
        print('Mindataset len='+str(mini_dataset_len))
        self.dataset_len=mini_dataset_len*self.noise_repeat_factor

        # create list of indexes for non-iidness
        list_of_idxs=[]
        for i in range(0,self.splits):
            idxs=[]
            list_of_idxs.append(idxs)

        idx=0
        for batch_idx, (data, target) in enumerate(current_trainset):
            list_of_idxs[target%self.splits].append(idx)
            idx=idx+1

        # create list of same-indexed subdatasets
        sublist=[]
        for idxs in list_of_idxs:
            sub=torch.utils.data.Subset(current_trainset, idxs)
            sublist.append(sub)
            print(len(idxs))
        train_loader_list=[]
        for sub in sublist:
            temp_train_loader = torch.utils.data.DataLoader(sub, self.batch_size, shuffle=False)
            train_loader_list.append(temp_train_loader)

        # non-eed
        # calculate minimum batch
        mymin=1
        print('Mymin='+str(mymin))

        # eed
        D=mini_dataset_len/self.batch_size # dataset size in batches
        print('D (batches)='+str(D))
        mu = D / self.subdatasetnum # average per client dataset
        print('mu (batches)='+str(mu))
        sigma = mu * self.sd  # sigma in relation to mu
        subdataset_batch_list = []  # final dataset list

        # apply gaussian distribution to eed, with N(mu,sigma)
        new_size_list=np.random.zipf(self.sd, self.subdatasetnum)
        sum = 0
        for i in new_size_list:
            sum = sum + i
        new_size_list = [x / sum for x in new_size_list]

        for i in range(0, self.subdatasetnum):
            current_list_size = get_list_sum(subdataset_batch_list)
            new_size = round(new_size_list[i]*D)
            if current_list_size + new_size <= D:
                subdataset_batch_list.append(new_size)
            else:
                subdataset_batch_list.append(max(round(D - current_list_size),mymin))
        # sort dataset list
        subdataset_batch_list=sorted(subdataset_batch_list)

        # in case total dataset list!=D, uniformly add difference
        if get_list_sum(subdataset_batch_list)==D:
            print('Mydiff=0, no need for normalization')
        elif get_list_sum(subdataset_batch_list)<D:
            print('Mydiff is negative, I have added less data, need to uniformly add')
            while get_list_sum(subdataset_batch_list)<D:
                i=0
                while get_list_sum(subdataset_batch_list)<D and i<len(subdataset_batch_list):
                    subdataset_batch_list[i]=subdataset_batch_list[i]+mymin
                    i=i+1
            print('My new diff is='+str(get_list_sum(subdataset_batch_list)-D))
        elif get_list_sum(subdataset_batch_list)>D:
            print('Mydiff is positive, I have added more data, need to uniformly remove')
            while get_list_sum(subdataset_batch_list)>D:
                i=0
                while get_list_sum(subdataset_batch_list)>D and i<len(subdataset_batch_list):
                    if subdataset_batch_list[i]>mymin:
                        subdataset_batch_list[i]=subdataset_batch_list[i]-mymin
                    i=i+1
            print('My new diff is='+str(get_list_sum(subdataset_batch_list)-D))

        # normalize dataset list to not contain zero values
        for i in range(0,self.subdatasetnum):
            if subdataset_batch_list[i]<mymin:
                diff=mymin-subdataset_batch_list[i]
                subdataset_batch_list[i] = mymin
                j = 0
                found = False
                while j < self.subdatasetnum and (not found):
                    if subdataset_batch_list[-j] > 2 * mymin:
                        subdataset_batch_list[-j] = subdataset_batch_list[-j] - diff
                        found = True
                    j = j + 1

        # unsort/shuffle dataset list
        random.shuffle(subdataset_batch_list)

        print('Batch list='+str(subdataset_batch_list))
        print('with len='+str(len(subdataset_batch_list)))
        sum=0
        for el in subdataset_batch_list:
            sum=sum+el
        print('and total sum='+str(sum))

        #subdataset_batches=mini_dataset_len/(self.batch_size*self.subdatasetnum)
        unique_id=0
        subdataset = Subdataset()
        for load in train_loader_list:
            for batch_idx, (data, target) in enumerate(load):
                #print('Unique_id='+str(unique_id)+'batchh='+str(len(subdataset.lst)))
                if unique_id<self.subdatasetnum:
                    if len(subdataset.lst) < subdataset_batch_list[unique_id]:
                        subdataset.lst.append(((data, target)))
                    else:
                        self.subdatasets.add_new(subdataset)
                        unique_id=unique_id+1
                        subdataset=Subdataset()
                        subdataset.lst.append(((data, target)))
                else:
                    # again due to truncacion we add all remainders (few batches) to the last record
                    print('Adding extra trunc ' + str(len(subdataset.lst)))
                    subdataset.lst.append(((data, target)))
        self.subdatasets.add_new(subdataset)

        print('Before removal='+str(len(self.subdatasets.db)))
        #remove subdatasets if more than subdataset num due to truncactions
        self.subdatasets.remove_one(self.subdatasetnum)
        print('Found a total of subdatasets='+str(len(self.subdatasets.db)))

        # debug print results
        mystr=''
        mysum=0
        for sub in self.subdatasets.db:
            mystr=mystr+str(len(sub.lst))+','
            mysum=mysum+len(sub.lst)
        print('Reset subdatasets output= '+mystr+', with size='+str(mysum))
        print('Total='+str(mysum))

        # non-iid shuffle
        random.shuffle(self.subdatasets.db)

    def create_dataset(self):
        transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                 AddGaussianNoise(0., 0.0001)])

        train_set = torchvision.datasets.SVHN(root=Global._ROOT + Global._DATASETS_FOLDER + Global._SVHN_PATH,
                                                  split='extra', transform=transform, target_transform=None,
                                                  download=True)
        return train_set

    def is_dataset_depleted(self):
        for sub in self.subdatasets.db:
            if not sub.is_used:
                return False
        return True

    def is_dataset_depleted_with_noise(self):
        return (self.noise_repeat_factor<=self.noise_repeats)

    def unmark_used_subdataset(self,id):
        found=False
        for sub in self.subdatasets.db:
            if sub.id==id:
                found=True
                # we only care for this noise round's datasets
                if sub.is_selected:
                    sub.is_used=False
        if not found:
            print(str('debug ERROR used subdataset not found!'))

    def mark_used_subdataset(self,id):
        found=False
        for sub in self.subdatasets.db:
            if sub.id==id:
                found = True
                # we only care for this noise round's datasets
                if sub.is_selected:
                    sub.is_used=True
        if not found:
            print(str('debug ERROR used subdataset not found!'))

    def debug_subdatasets(self):
        mylist=[]
        for sub in self.subdatasets.db:
            mystr=str(sub.id)+'-'+str(sub.is_selected)+'-'+str(sub.is_used)
            mylist.append(mystr)
        return mylist

    def unmark_all_selected(self):
        for sub in self.subdatasets.db:
            sub.is_selected=False

    def mark_selected_subdataset(self,id):
        found=False
        for sub in self.subdatasets.db:
            if sub.id==id:
                sub.is_selected=True
                found=True
        if not found:
            print(str('debug ERROR used subdataset not found!'))

    def unmark_selected_subdataset(self,id):
        found=False
        for sub in self.subdatasets.db:
            if sub.id==id:
                sub.is_selected=False
                found=True
        if not found:
            print(str('debug ERROR used subdataset not found!'))

    def get_next_subset(self):
        for sub in self.subdatasets.db:
            if (not sub.is_used) and (not sub.is_selected):
                return sub
        return None

    def get_next_subset_repeat(self):
        sub_to_return=None
        for sub in self.subdatasets.db:
            if (not sub.is_used) and (not sub.is_selected):
                sub_to_return=sub
        if sub_to_return is None: #reset datasets
            self.noise_repeats = self.noise_repeats+1
            if self.is_dataset_depleted_with_noise():
                sub_to_return=None
            else:
                self.reset_subdatasets()
                sub_to_return=self.get_next_subset_repeat()
        return sub_to_return

    def get_list_of_all_datasets(self):
        mylist=[]
        for sub in self.subdatasets.db:
            mylist.append(sub.id)
        mylist=sorted(mylist)
        return mylist

    def get_list_of_used_datasets(self):
        mylist=[]
        for sub in self.subdatasets.db:
            if sub.is_used:
                mylist.append(sub.id)
        mylist=sorted(mylist)
        return mylist

    def do_test(self,model):
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        model.eval()
        test_loss = 0
        correct = 0
        actuals = []
        predictions = []
        with torch.no_grad():
            for data, target in self.test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.argmax(1, keepdim=True)
                actuals.extend(target.view_as(pred))
                predictions.extend(pred)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        torch.save(model.state_dict(), Global._ROOT+Global._MODELS_FOLDER+'SVHN.pt')
        #conf=confusion_matrix(actuals, predictions, labels=labels)
        return test_loss, float(100 * correct / len(self.test_loader.dataset))

class SVHN_big_noniid_bipolar:
    def __init__(self,subdatasetnum,mybatch,mylearnrate,splits,sd,percent_iid):
        self.learning_rate=mylearnrate
        self.subdatasetnum=subdatasetnum
        self.subdatasets=Subdatasets()
        self.batch_size=mybatch
        self.noise_repeat_factor=1
        self.noise_repeats=0
        self.splits=splits
        self.sd=sd
        self.percent_iid=percent_iid
        print('SVHN big with batch-rate= '+str(mybatch)+'-'+str(mylearnrate))

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_set=torchvision.datasets.SVHN(root=Global._ROOT + Global._DATASETS_FOLDER + Global._SVHN_PATH,
                                  split='test', transform=transform, target_transform=None, download=True)

        self.test_loader = torch.utils.data.DataLoader(test_set, self.batch_size, shuffle=True)
        self.dataset_size=self.noise_repeat_factor*Tools.get_file_size_in_mb(Global._ROOT+Global._DATASETS_FOLDER+Global._SVHN_DATASET_PATH)
        self.dataset_len=0
        self.reset_subdatasets()

    def reset_subdatasets(self):
        self.subdatasets=Subdatasets()
        current_trainset=self.create_dataset()

        iid_size=int(len(current_trainset)*self.percent_iid)
        noniid_size = len(current_trainset) -iid_size
        iid_subset, noniid_subset = random_split(current_trainset, [iid_size, noniid_size], generator=torch.Generator().manual_seed(2))

        mini_dataset_len=len(iid_subset)+len(noniid_subset)
        print('Mindataset len='+str(mini_dataset_len))
        self.dataset_len=mini_dataset_len*self.noise_repeat_factor

        # create list of indexes for non-iidness
        list_of_idxs=[]
        for i in range(0,self.splits):
            idxs=[]
            list_of_idxs.append(idxs)

        idx=0
        for batch_idx, (data, target) in enumerate(noniid_subset):
            list_of_idxs[target%self.splits].append(idx)
            idx=idx+1

        # create list of same-indexed subdatasets
        sublist=[]
        for idxs in list_of_idxs:
            sub=torch.utils.data.Subset(noniid_subset, idxs)
            sublist.append(sub)
            print(len(idxs))
        train_loader_list=[]
        for sub in sublist:
            temp_train_loader = torch.utils.data.DataLoader(sub, self.batch_size, shuffle=False)
            train_loader_list.append(temp_train_loader)
        temp_train_loader_iid=torch.utils.data.DataLoader(iid_subset, self.batch_size, shuffle=False)
        train_loader_list.append(temp_train_loader_iid)

        # non-eed
        # calculate minimum batch
        mymin=1
        print('Mymin='+str(mymin))

        # eed
        D=mini_dataset_len/self.batch_size # dataset size in batches
        print('D (batches)='+str(D))
        mu = D / self.subdatasetnum # average per client dataset
        print('mu (batches)='+str(mu))
        sigma = mu * self.sd  # sigma in relation to mu
        subdataset_batch_list = []  # final dataset list

        # apply gaussian distribution to eed, with N(mu,sigma)
        new_size_list=np.random.zipf(self.sd, self.subdatasetnum)
        sum = 0
        for i in new_size_list:
            sum = sum + i
        new_size_list = [x / sum for x in new_size_list]

        for i in range(0, self.subdatasetnum):
            current_list_size = get_list_sum(subdataset_batch_list)
            new_size = round(new_size_list[i]*D)
            if current_list_size + new_size <= D:
                subdataset_batch_list.append(new_size)
            else:
                subdataset_batch_list.append(max(round(D - current_list_size),mymin))
        # sort dataset list
        subdataset_batch_list=sorted(subdataset_batch_list)

        # in case total dataset list!=D, uniformly add difference
        if get_list_sum(subdataset_batch_list)==D:
            print('Mydiff=0, no need for normalization')
        elif get_list_sum(subdataset_batch_list)<D:
            print('Mydiff is negative, I have added less data, need to uniformly add')
            while get_list_sum(subdataset_batch_list)<D:
                i=0
                while get_list_sum(subdataset_batch_list)<D and i<len(subdataset_batch_list):
                    subdataset_batch_list[i]=subdataset_batch_list[i]+mymin
                    i=i+1
            print('My new diff is='+str(get_list_sum(subdataset_batch_list)-D))
        elif get_list_sum(subdataset_batch_list)>D:
            print('Mydiff is positive, I have added more data, need to uniformly remove')
            while get_list_sum(subdataset_batch_list)>D:
                i=0
                while get_list_sum(subdataset_batch_list)>D and i<len(subdataset_batch_list):
                    if subdataset_batch_list[i]>mymin:
                        subdataset_batch_list[i]=subdataset_batch_list[i]-mymin
                    i=i+1
            print('My new diff is='+str(get_list_sum(subdataset_batch_list)-D))

        # normalize dataset list to not contain zero values
        for i in range(0,self.subdatasetnum):
            if subdataset_batch_list[i]<mymin:
                diff=mymin-subdataset_batch_list[i]
                subdataset_batch_list[i] = mymin
                j = 0
                found = False
                while j < self.subdatasetnum and (not found):
                    if subdataset_batch_list[-j] > 2 * mymin:
                        subdataset_batch_list[-j] = subdataset_batch_list[-j] - diff
                        found = True
                    j = j + 1

        # unsort/shuffle dataset list
        random.shuffle(subdataset_batch_list)

        print('Batch list='+str(subdataset_batch_list))
        print('with len='+str(len(subdataset_batch_list)))
        sum=0
        for el in subdataset_batch_list:
            sum=sum+el
        print('and total sum='+str(sum))

        #subdataset_batches=mini_dataset_len/(self.batch_size*self.subdatasetnum)
        unique_id=0
        subdataset = Subdataset()
        for load in train_loader_list:
            for batch_idx, (data, target) in enumerate(load):
                #print('Unique_id='+str(unique_id)+'batchh='+str(len(subdataset.lst)))
                # workaround to not include lftover batches with only one np.array (tensor flow break)
                if len(target)<2:
                    pass
                else:
                    if unique_id<self.subdatasetnum:
                        if len(subdataset.lst) < subdataset_batch_list[unique_id]:
                            subdataset.lst.append(((data, target)))
                        else:
                            self.subdatasets.add_new(subdataset)
                            unique_id=unique_id+1
                            subdataset=Subdataset()
                            subdataset.lst.append(((data, target)))
                    else:
                        # again due to truncacion we add all remainders (few batches) to the last record
                        print('Adding extra trunc ' + str(len(subdataset.lst)))
                        subdataset.lst.append(((data, target)))
        #self.subdatasets.add_new(subdataset)

        print('Before removal='+str(len(self.subdatasets.db)))
        #remove subdatasets if more than subdataset num due to truncactions
        #self.subdatasets.remove_one(self.subdatasetnum)
        print('Found a total of subdatasets='+str(len(self.subdatasets.db)))

        # debug print results
        mystr=''
        mysum=0
        for sub in self.subdatasets.db:
            mystr=mystr+str(len(sub.lst))+','
            mysum=mysum+len(sub.lst)
        print('Reset subdatasets output= '+mystr+', with size='+str(mysum))
        print('Total='+str(mysum))

        # non-iid shuffle
        random.shuffle(self.subdatasets.db)

    def print_all_dataset_classes(self):
        for sub in self.subdatasets.db:
            print('Id:'+str(sub.id)+',cl:'+str(sub.get_total_classes()))

    def create_dataset(self):
        transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                 AddGaussianNoise(0., 0.0001)])

        train_set = torchvision.datasets.SVHN(root=Global._ROOT + Global._DATASETS_FOLDER + Global._SVHN_PATH,
                                                  split='extra', transform=transform, target_transform=None,
                                                  download=True)
        return train_set

    def is_dataset_depleted(self):
        for sub in self.subdatasets.db:
            if not sub.is_used:
                return False
        return True

    def is_dataset_depleted_with_noise(self):
        return (self.noise_repeat_factor<=self.noise_repeats)

    def unmark_used_subdataset(self,id):
        found=False
        for sub in self.subdatasets.db:
            if sub.id==id:
                found=True
                # we only care for this noise round's datasets
                if sub.is_selected:
                    sub.is_used=False
        if not found:
            print(str('debug ERROR used subdataset not found!'))

    def mark_used_subdataset(self,id):
        found=False
        for sub in self.subdatasets.db:
            if sub.id==id:
                found = True
                # we only care for this noise round's datasets
                if sub.is_selected:
                    sub.is_used=True
        if not found:
            print(str('debug ERROR used subdataset not found!'))

    def debug_subdatasets(self):
        mylist=[]
        for sub in self.subdatasets.db:
            mystr=str(sub.id)+'-'+str(sub.is_selected)+'-'+str(sub.is_used)
            mylist.append(mystr)
        return mylist

    def unmark_all_selected(self):
        for sub in self.subdatasets.db:
            sub.is_selected=False

    def mark_selected_subdataset(self,id):
        found=False
        for sub in self.subdatasets.db:
            if sub.id==id:
                sub.is_selected=True
                found=True
        if not found:
            print(str('debug ERROR used subdataset not found!'))

    def unmark_selected_subdataset(self,id):
        found=False
        for sub in self.subdatasets.db:
            if sub.id==id:
                sub.is_selected=False
                found=True
        if not found:
            print(str('debug ERROR used subdataset not found!'))

    def get_next_subset(self):
        for sub in self.subdatasets.db:
            if (not sub.is_used) and (not sub.is_selected):
                return sub
        return None

    def get_next_subset_repeat(self):
        sub_to_return=None
        for sub in self.subdatasets.db:
            if (not sub.is_used) and (not sub.is_selected):
                sub_to_return=sub
        if sub_to_return is None: #reset datasets
            self.noise_repeats = self.noise_repeats+1
            if self.is_dataset_depleted_with_noise():
                sub_to_return=None
            else:
                self.reset_subdatasets()
                sub_to_return=self.get_next_subset_repeat()
        return sub_to_return

    def get_list_of_all_datasets(self):
        mylist=[]
        for sub in self.subdatasets.db:
            mylist.append(sub.id)
        mylist=sorted(mylist)
        return mylist

    def get_list_of_used_datasets(self):
        mylist=[]
        for sub in self.subdatasets.db:
            if sub.is_used:
                mylist.append(sub.id)
        mylist=sorted(mylist)
        return mylist

    def do_test(self,model):
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        model.eval()
        test_loss = 0
        correct = 0
        actuals = []
        predictions = []
        with torch.no_grad():
            for data, target in self.test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.argmax(1, keepdim=True)
                actuals.extend(target.view_as(pred))
                predictions.extend(pred)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        torch.save(model.state_dict(), Global._ROOT+Global._MODELS_FOLDER+'SVHN.pt')
        #conf=confusion_matrix(actuals, predictions, labels=labels)
        return test_loss, float(100 * correct / len(self.test_loader.dataset))



def test_split():
#0,0.1,1,3 kai sto noniid    def __init__(self,subdatasetnum,mybatch,mylearnrate,splits,sd):
    test=SVHN_big(100,128,0.1,1)
    idx=0
    for sub in test.subdatasets:
        sub_str=''
        for batch_idx, (data, target) in enumerate(sub.lst):
            sub_str=sub_str+str(target)
        with open(Global._ROOT + Global._LOGS_FOLDER + 'test', mode='a') as file:
            file.write(sub_str + '\n')
            file.write('Subdataset finito sub=' + str(idx))
        idx=idx+1

def test_iid():
#0,0.1,1,3 kai sto noniid
    test = SVHN_big_noniid(100, 128, 0.1, 3, 1000)
    idx=0
    for sub in test.subdatasets.db:
        sub_str=''
        for batch_idx, (data, target) in enumerate(sub.lst):
            sub_str=sub_str+str(target)
        with open(Global._ROOT + Global._LOGS_FOLDER + 'test', mode='a') as file:
            file.write(sub_str + '\n')
            file.write('Subdataset finito sub=' + str(idx))
        idx=idx+1

#test_iid()