TRAIN_POWER_CONSUMPTION=50 # watt
TRAIN_SPEED=6000# samples/sec

class Mecs:
    db=[]

    def add_new_mec(self,id,worker):
        new_mec=Mec(id,worker)
        self.db.append(new_mec)

    def get_all_mec_ids(self):
        mylist=[]
        for mec in self.db:
            mylist.append(mec.id)
        return mylist

    def get_mec_list(self):
        mylist=[]
        for mec in self.db:
            mylist.append(mec)
        return mylist

    def get_mec_from_id(self,id):
        for mec in self.db:
            if mec.id==id:
                return mec

    def set_active(self,id):
        found=False
        for mec in self.db:
            if mec.id==id:
                mec.is_active=True
                found=True
                break
        if not found:
            print(str('Error - cant find set_active mec id'))

    def get_mecs_without_models(self):
        mylist=[]
        for mec in self.db:
            if not mec.owns_model:
                mylist.append(mec)
        return mylist

    def get_active_mecs(self):
        mylist=[]
        for mec in self.db:
            if mec.is_active:
                mylist.append(mec)
        return mylist

    def get_active_mecs_workers(self):
        mylist=[]
        for mec in self.db:
            if mec.is_active:
                mylist.append(mec.worker)
        return mylist

    def kill_all_workers(self):
        for node in self.db:
            if node.worker is not None:
                node.worker.clear_objects()
            #node.worker=None
            node.dataset = None
            node.model=None
            node.optimizer = None
            node.loss = None
            node.params = []

    def kill_worker(self,mec_id):
        for node in self.db:
            if node.id==mec_id:
                if node.worker is not None:
                    node.worker.clear_objects()
                #node.worker=None
                node.dataset = None
                node.model=None
                node.optimizer = None
                node.loss = None
                node.params = []

    def clear_all_selected(self):
        for node in self.db:
            node.is_active=False

    def clean(self):
        self.clear_all_selected()
        self.kill_all_workers()

    def clean_inactive(self):
        for mec in self.db:
            if not mec.is_active:
                self.kill_worker(mec.id)

    def calc_total_energy_spent_proc(self):
        total=0
        for node in self.db:
            total=total+node.energyPROC
        return total

    def calc_total_time_spent_proc(self):
        total=0
        for node in self.db:
            total=total+node.timePROC
        return total

    def calc_total_mb_UL(self):
        total=0
        for node in self.db:
            total=total+node.mbUL
        return total

    def calc_total_mb_DL(self):
        total=0
        for node in self.db:
            total=total+node.mbDL
        return total



class Mec:
    def __init__(self,id,worker):
        self.id=id
        self.worker=worker
        self.dataset=[]
        self.dataset_size=0
        self.datasetid=None
        self.model = None
        self.optimizer = None
        self.loss = None
        self.params=[]
        self.energyPROC = 0
        self.timePROC=0
        self.mbUL=0
        self.mbDL=0
        self.is_active=False

    def update_dataset_per_client_size(self, new_info_len,total_dataset_len, total_dataset_size):
        self.dataset_size=self.dataset_size+total_dataset_size * (new_info_len / total_dataset_len)

    def calc_metrics_training(self,batch_size):
        time_to_train=self.calc_time_training(batch_size)
        self.timePROC=self.timePROC+time_to_train
        self.energyPROC=self.energyPROC+self.calc_energy_training(time_to_train)

    def calc_energy_training(self,time):
        return time*TRAIN_POWER_CONSUMPTION

    def calc_time_training(self,batch_size):
        samples = len(self.dataset) * batch_size
        return float(samples / TRAIN_SPEED)


