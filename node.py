import Global
class Nodes:
    db=[]

    def add_new(self,id,name,worker):
        new_node=Node(name,worker,id)
        self.db.append(new_node)

    def add_new_with_dataset(self,id,name,worker,dataset):
        new_node=Node(name,worker,id)
        new_node.dataset=dataset
        self.db.append(new_node)

    def get_worker_from_name(self,name):
        for node in self.db:
            if node.name==name:
                return node.worker

    def get_node_from_id(self,id):
        for node in self.db:
            if node.id==id:
                return node

    def get_id_from_name(self,name):
        for node in self.db:
            if node.name==name:
                return node.id

    def update_initially_selected_clients(self,total_selected_list):
        for node in self.db:
            if node in total_selected_list:
                node.is_selected=True

    def get_available_list(self,total_online_list):
        available_list=[]
        for node in self.db:
            if (node.name in total_online_list) and (node.was_chosen_before==False):
                available_list.append(node)
        return available_list

    def get_selected_list(self):
        selected_list=[]
        for node in self.db:
            if node.is_selected:
                selected_list.append(node)
        return selected_list

    def get_selected_list_per_mec(self,mec_id):
        selected_list=[]
        for node in self.db:
            if (node.is_selected) and (node.mec_id==mec_id):
                selected_list.append(node)
        return selected_list

    def get_selected_list_workers(self):
        selected_list=[]
        for node in self.db:
            if node.is_selected:
                selected_list.append(node.worker)
        return selected_list

    def get_total_selected_(self):
        selected_list=[]
        for node in self.db:
            if node.is_selected:
                selected_list.append(node)
        return len(selected_list)

    def get_total(self):
        return len(self.db)

    def get_dataset_list_from_selected_clients(self):
        datasets=[]
        for node in self.db:
            if node.is_selected:
                datasets.append(node.dataset)
        return datasets

    def get_dataset_list_from_selected_clients_per_mec(self,mec_id):
        datasets=[]
        for node in self.db:
            if node.is_selected and node.mec_id==mec_id:
                datasets.append(node.dataset)
        return datasets

    def get_per_user_dataset_list(self):
        mylist=[]
        for node in self.db:
            mylist.append((node.id,node.dataset_size))
        return mylist

    def calc_total_energy_spent_UL_succ(self):
        total_energy=0
        for node in self.db:
            total_energy=total_energy+node.energyUL_succ
        return total_energy

    def calc_total_energy_spent_DL_succ(self):
        total_energy=0
        for node in self.db:
            total_energy=total_energy+node.energyDL_succ
        return total_energy

    def calc_total_energy_spent_proc_succ(self):
        total_energy=0
        for node in self.db:
            total_energy=total_energy+node.energyPROC_succ
        return total_energy

    def calc_total_energy_spent_UL_fail(self):
        total_energy=0
        for node in self.db:
            total_energy=total_energy+node.energyUL_fail
        return total_energy

    def calc_total_energy_spent_DL_fail(self):
        total_energy=0
        for node in self.db:
            total_energy=total_energy+node.energyDL_fail
        return total_energy

    def calc_total_energy_spent_proc_fail(self):
        total_energy=0
        for node in self.db:
            total_energy=total_energy+node.energyPROC_fail
        return total_energy

    def calc_total_mb_UL(self):
        total_mb=0
        for node in self.db:
            total_mb=total_mb+node.mbUL
        return total_mb

    def calc_total_mb_DL(self):
        total_mb=0
        for node in self.db:
            total_mb=total_mb+node.mbDL
        return total_mb

    def calc_total_mb_lost_UL(self):
        total_mb=0
        for node in self.db:
            total_mb=total_mb+node.mblostUL
        return total_mb

    def calc_total_mb_lost_DL(self):
        total_mb=0
        for node in self.db:
            total_mb=total_mb+node.mblostDL
        return total_mb


    def calc_total_time_spent_UL(self):
        outt=0
        for node in self.db:
            outt=outt+node.timeUL
        return outt

    def calc_total_time_spent_DL(self):
        outt=0
        for node in self.db:
            outt=outt+node.timeDL
        return outt

    def calc_total_time_spent_proc(self):
        outt=0
        for node in self.db:
            outt=outt+node.timePROC
        return outt



    def debug_print_datset_per_client_size(self):
        sum=0
        for node in self.db:
            print(str(node.dataset_size))
            sum=sum+node.dataset_size
        print('Total datasets size=' + str(sum))

    def kill_all_workers(self):
        for node in self.db:
            if node.worker is not None:
                node.worker.clear_objects()
            node.worker=None
            self.dataset = None

    def clear_all_selected(self):
        for node in self.db:
            node.is_selected=False

    def clean(self):
        self.clear_all_selected()
        self.kill_all_workers()


    def clear_all_chosen_before(self):
        for node in self.get_selected_list():
            node.was_chosen_before=False

    def set_node_selection(self,id,mybool):
        found=False
        for node in self.db:
            if node.id==id:
                node.is_selected=mybool
                found=True
                break
        if (not found):
            Global.mydebug('(ERROR): Cannot find node for selection'+str(id))

    def set_node_chosen_before(self,id,mybool):
        found=False
        for node in self.db:
            if node.id==id:
                node.was_chosen_before=mybool
                found=True
                break
        if (not found):
            Global.mydebug('(ERROR): Cannot find node for selection'+str(id))

    def add_node_mblostUL(self,id,mb):
        found=False
        for node in self.db:
            if node.id==id:
                node.mblostUL=node.mblostUL+mb
                found=True
                break
        if (not found):
            Global.mydebug('(ERROR): Cannot find node for add mblostUL'+str(id))

    def add_node_mblostDL(self,id,mb):
        found=False
        for node in self.db:
            if node.id==id:
                node.mblostDL=node.mblostDL+mb
                found=True
                break
        if (not found):
            Global.mydebug('(ERROR): Cannot find node for add mblostDL'+str(id))

    def add_node_mbUL(self,id,mb):
        found=False
        for node in self.db:
            if node.id==id:
                node.mbUL=node.mbUL+mb
                found=True
                break
        if (not found):
            Global.mydebug('(ERROR): Cannot find node for add mblostUL'+str(id))

    def add_node_mbDL(self,id,mb):
        found=False
        for node in self.db:
            if node.id==id:
                node.mbDL=node.mbDL+mb
                found=True
                break
        if (not found):
            Global.mydebug('(ERROR): Cannot find node for add mblostDL'+str(id))

    def add_node_energyUL_succ(self,id,energy):
        found=False
        for node in self.db:
            if node.id==id:
                node.energyUL_succ=node.energyUL_succ+energy
                found=True
                break
        if (not found):
            Global.mydebug('(ERROR): Cannot find node for add energyUL'+str(id))

    def add_node_energyPROC_temp(self,id,energy):
        found=False
        for node in self.db:
            if node.id==id:
                node.energyPROC_temp=energy
                found=True
                break
        if (not found):
            Global.mydebug('(ERROR): Cannot find node for add energyTemp'+str(id))

    def add_node_energyPROC_succ(self,id):
        found=False
        for node in self.db:
            if node.id==id:
                node.energyPROC_succ=node.energyPROC_succ+node.energyPROC_temp
                found=True
                break
        if (not found):
            Global.mydebug('(ERROR): Cannot find node for add energyPROC'+str(id))

    def add_node_energyDL_succ(self,id,energy):
        found=False
        for node in self.db:
            if node.id==id:
                node.energyDL_succ=node.energyDL_succ+energy
                found=True
                break
        if (not found):
            Global.mydebug('(ERROR): Cannot find node for add energyDL'+str(id))

    def add_node_energyUL_fail(self,id,energy):
        found=False
        for node in self.db:
            if node.id==id:
                node.energyUL_fail=node.energyUL_fail+energy
                found=True
                break
        if (not found):
            Global.mydebug('(ERROR): Cannot find node for add energyUL'+str(id))

    def add_node_energyPROC_fail(self,id):
        found=False
        for node in self.db:
            if node.id==id:
                node.energyPROC_fail=node.energyPROC_fail+node.energyPROC_temp
                found=True
                break
        if (not found):
            Global.mydebug('(ERROR): Cannot find node for add energyPROC'+str(id))

    def add_node_energyDL_fail(self,id,energy):
        found=False
        for node in self.db:
            if node.id==id:
                node.energyDL_fail=node.energyDL_fail+energy
                found=True
                break
        if (not found):
            Global.mydebug('(ERROR): Cannot find node for add energyDL'+str(id))
    def add_node_timePROC(self,id,time):
        found=False
        for node in self.db:
            if node.id==id:
                node.timePROC=node.timePROC+time
                found=True
                break
        if (not found):
            Global.mydebug('(ERROR): Cannot find node for add timePROC'+str(id))

    def add_node_timeUL(self,id,time):
        found=False
        for node in self.db:
            if node.id==id:
                node.timeUL=node.timeUL+time
                found=True
                break
        if (not found):
            Global.mydebug('(ERROR): Cannot find node for add energyUL'+str(id))

    def add_node_timeDL(self,id,time):
        found=False
        for node in self.db:
            if node.id==id:
                node.timeDL=node.timeDL+time
                found=True
                break
        if (not found):
            Global.mydebug('(ERROR): Cannot find node for add energyDL'+str(id))

class Node:
    def __init__(self,name,worker,id):
        self.id=id
        self.worker=worker
        self.name=name
        self.dataset=[]
        self.dataset_size=0
        self.datasetid=None
        self.model = None
        self.optimizer = None
        self.loss = None
        self.params=[]
        self.is_selected=False
        self.was_chosen_before=False
        self.energyUL_succ=0
        self.energyDL_succ=0
        self.energyPROC_succ = 0
        self.energyUL_fail=0
        self.energyDL_fail=0
        self.energyPROC_fail = 0
        self.energyPROC_temp=0
        self.timeUL=0
        self.timeDL=0
        self.timePROC=0
        self.mbUL=0
        self.mbDL=0
        self.mblostUL=0
        self.mblostDL=0
        self.mec_id=None


    def update_dataset_per_client_size(self, new_info_len,total_dataset_len, total_dataset_size):
        self.dataset_size=self.dataset_size+total_dataset_size * (new_info_len / total_dataset_len)
