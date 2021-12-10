import gc
import random
import math
import sys
import torch
import torch.utils.data as torch_data
import torch.nn.functional as F
import syft as sy  # <-- NEW: import the Pysyft library
import Global
import time
import datetime
from torch import optim

import Tools
from model import SVHN_Model
from node import Nodes
from learning import SVHN_big_noniid,SVHN_big
from mobility import Mobility_Shanghai, Mobility_Wifi
from access import Access_LTE, Access_WIFI
from energy import Energy_LTE, Energy_Wifi, Energy_CPU_DRL, Energy_CPU_Inception
from core import Core_LTE, Core_WIFI, Core_LTE_edge_regional, Core_LTE_edge_access
from cloud import Cloud
from mec import Mecs

class Experiment:
    def __init__(self,_DATE_A,_DATE_B,_DATE_C,_T_BEGIN,_T_END,_SUBDATASETS,_DEVICES_PER_ROUND,
         _IID,_EED,_WIRELESS,_MODEL,_EXPORT_CSV,_USE_LOGGER,_ML,_AGGREGATION_FACTOR,_WAITING_FACTOR_CL,
                 _LOCAL_EPOCHS,_BATCH_SIZE,_LEARNING_RATE):

        self._DATE_A=int(_DATE_A)
        self._DATE_B=int(_DATE_B)
        self._DATE_C=int(_DATE_C)
        self._DATE=[int(_DATE_A), int(_DATE_B), int(_DATE_C)]
        self._T_BEGIN=int(_T_BEGIN)
        self._T_END=int(_T_END)
        self._SUBDATASETS=int(_SUBDATASETS)
        self._DEVICES_PER_ROUND=int(_DEVICES_PER_ROUND)
        self._IID=int(_IID)
        self._EED=float(_EED)
        self._WIRELESS=str(_WIRELESS)
        self._MODEL=str(_MODEL)
        self._EXPORT_CSV=bool(_EXPORT_CSV)
        self._USE_LOGGER=bool(_USE_LOGGER)
        self._ML= str(_ML)
        self._AGGREGATION_FACTOR=int(_AGGREGATION_FACTOR)
        self._WAITING_FACTOR_CL=int(_WAITING_FACTOR_CL)
        self._LOCAL_EPOCHS=int(_LOCAL_EPOCHS)
        self._BATCH_SIZE=int(_BATCH_SIZE)
        self._LEARNING_RATE=float(_LEARNING_RATE)
        Global._DEBUG_FILENAME=str(datetime.datetime.now()).replace(':','-')+'_'+str(self._ML)
        self.hook = sy.TorchHook(torch)
        self.crypto_provider = sy.VirtualWorker(self.hook, id="crypto_provider")
        self.active_learners=[]
        self.open_logger()
        self.debug_param_info()
        self.merged_dataset=[]
        self.has_begun_superbatch_training=False

        if self._MODEL == 'SVHN':
            self.central_model = SVHN_Model()
            if self._IID == -1:
                self.learning = SVHN_big(self._SUBDATASETS,self._BATCH_SIZE,self._LEARNING_RATE,self._EED)
            if self._IID == 0:
                self.learning = SVHN_big(self._SUBDATASETS,self._BATCH_SIZE,self._LEARNING_RATE,self._EED)
            elif self._IID == 3:
                self.learning = SVHN_big_noniid(self._SUBDATASETS,self._BATCH_SIZE,self._LEARNING_RATE,splits=3,sd=self._EED)
            elif self._IID == 5:
                self.learning = SVHN_big_noniid(self._SUBDATASETS,self._BATCH_SIZE,self._LEARNING_RATE,splits=5,sd=self._EED)
            elif self._IID == 7:
                self.learning = SVHN_big_noniid(self._SUBDATASETS,self._BATCH_SIZE,self._LEARNING_RATE,splits=7,sd=self._EED)
            self.energy_proc = Energy_CPU_Inception()
            self.cloud = Cloud()
            Global.mydebug(str(datetime.datetime.now()) + ' (debug0) I have used datasets='+str(self.learning.get_list_of_all_datasets()))
            Global.mydebug(str(datetime.datetime.now()) + ' (debug0) with num=='+str(len(self.learning.get_list_of_all_datasets())))

        self.central_optimizer = optim.SGD(params=self.central_model.parameters(), lr=self.learning.learning_rate)
        self.central_params = list(self.central_model.parameters())

        if self._WIRELESS == 'LTE':
            self.mecs=Mecs()
            for mec_id in range(0,self._AGGREGATION_FACTOR):
                worker = sy.VirtualWorker(self.hook, id='mec' + str(mec_id))
                self.mecs.add_new_mec(mec_id,worker)
            self.mobility = Mobility_Shanghai(math.inf, self._T_BEGIN, self._T_END,self._DATE,self.mecs.get_all_mec_ids())
            self.network = Access_LTE()
            self.energy_trx = Energy_LTE()
            if self._ML=='FL' or self._ML=='CL':
                self.core = Core_LTE(hops=3)
            elif self._ML=='EL':
                if self._AGGREGATION_FACTOR<20:
                    self.core = Core_LTE_edge_regional(hops=3)
                else:
                    self.core = Core_LTE_edge_access(hops=3)
        else:
            self.mecs=Mecs()
            for mec_id in range(0,self._AGGREGATION_FACTOR):
                worker = sy.VirtualWorker(self.hook, id='mec' + str(mec_id))
                self.mecs.add_new_mec(mec_id,worker)
            self.mobility = Mobility_Wifi(self._DATE, math.inf, self._T_BEGIN, self._T_END)
            self.network = Access_WIFI()
            self.energy_trx = Energy_Wifi()
            if self._ML=='FL' or self._ML=='CL':
                self.core = Core_WIFI(hops=3)
            elif self._ML=='EL':
                print('Error, not implemented yet!')

        self.devices = Nodes()
        self.phantom_unique_id = 0
        self.CURRENT_TIME = self._T_BEGIN
        self.new_unique_worker_id = 0
        self.myround = 0
        self.run_round()

    def open_logger(self):
        mystr='ml,'+'model,'+'wireless,'+'datea,'+'dateb,'+'datec,'\
              +'tbegin,'+'tend,'\
              +'iid,'+'eed,'+'roundev,'+'subdatasets,'+'aggfactor,'+'waiting_factor,'\
              +'epochs,'+'batchsize,'+'learningrate,'\
              +'timestamp,'+'loss,'+'acc,'\
              +'mbUL,'+'mbDL,'+'mblostUL,'+'mblostDL,'+'timeUL,'+'timeDL,'+'timePROC,'\
              +'energyclientULsucc,'+'energyclientDLsucc,'+'energyclientPROCsucc,'\
              +'energyclientULfail,'+'energyclientDLfail,'+'energyclientPROCfail,' \
              + 'energyedgePROC,' + 'timeedgePROC,' + 'mbedgeUL,'+ 'mbedgeDL,' \
              + 'energy_core_bs_to_edge,' + 'energy_core_edge_to_cloud,' + 'time_core_bs_to_edge,' + 'time_core_edge_to_cloud,' \
              + 'energy_core_edge_to_bs,' + 'energy_core_cloud_to_edge,' + 'time_core_edge_to_bs,' + 'time_core_cloud_to_edge,' \
              + 'timecloud,' + 'energycloud,' \
              + 'myround,' + 'learners'
        if self._EXPORT_CSV:
            with open(Global._ROOT + Global._LOGS_FOLDER + Global._END_CSV_NAME, mode='a') as file:
                file.write(mystr + '\n')
    def mydebug(self,mystr):
        with open(Global._ROOT+Global._LOGS_FOLDER+self.debug_file_name+".txt", mode='a') as file:
            file.write(mystr + '\n')

    def debug_param_info(self):
        mystr=' (debug) LETS START,Settings are:'+\
        '_MODEL=' + str(self._MODEL) + ' ,_WIRELESS=' + str(self._WIRELESS) + ' ,_IID=' + str(self._IID) + \
        ' ,_DEVICES_PER_ROUND=' +str(self._DEVICES_PER_ROUND) + ' ,_T_BEGIN=' \
        + str(self._T_BEGIN) + ' ,_T_END=' + str(self._T_END) + ' ,_DATE=' \
        + str(self._DATE) + ' ,_SUBDATASETS=' + str(self._SUBDATASETS) + ' ,_DATE=' + str(self._DATE) + ' ,_ML=' \
        + str(self._ML) + ' ,_AGGREGATION_FACTOR=' + str(self._AGGREGATION_FACTOR) +',_WAITING_FACTOR_CL' \
        + str(self._WAITING_FACTOR_CL)
        Global.mydebug(str(datetime.datetime.now())+ mystr)


    def clean(self):
        self.mecs.clean()
        self.devices.clean()
        if self.crypto_provider is not None:
            self.crypto_provider.clear_objects()

    def get_new_unique_worker_id(self):
        self.new_unique_worker_id=self.new_unique_worker_id+1
        return self.new_unique_worker_id-1

    def run_round(self):

        while ((self.CURRENT_TIME < self._T_END) and (not self.learning.is_dataset_depleted_with_noise())):

            # check online devices
            total_online_list = self.mobility.get_present_ids_now_filtered_by_pull(self.CURRENT_TIME)
            Global.mydebug('(debug) Starting new round=' + str(self.myround) + ',with online clients= ' + str(len(total_online_list)))

            # check if available clients after clean else move to next myround
            if len(total_online_list) < 2:
                self.clean()
                self.myround = self.myround + 1
                self.CURRENT_TIME = self.CURRENT_TIME + self.mobility.wait_for_clients_time
                Global.mydebug('(debug) Cannot find enough clients - waiting for mobility dataset granularity')
                continue

            # random client selection
            if len(total_online_list) <= self._DEVICES_PER_ROUND:
                total_selected_list = total_online_list
            else:
                total_selected_list = random.sample(total_online_list, self._DEVICES_PER_ROUND)
            Global.mydebug('(debug) Selected clients= ' + str(len(total_selected_list)))

            # data assignment
            #Global.mydebug('BEFOREEE Round='+str(self.myround))
            #Global.mydebug('Noise factor=' + str(self.learning.noise_repeats))
            #Global.mydebug(str(self.learning.debug_subdatasets()))

            for clientid in total_selected_list:
                subset = self.learning.get_next_subset_repeat()

                if subset is not None:
                    #Global.mydebug('getting subset id=' + str(subset.id))
                    name = clientid
                    id = self.get_new_unique_worker_id()
                    worker = sy.VirtualWorker(self.hook, id=name + str(id))
                    self.devices.add_new(id, name, worker)
                    lucky_node = self.devices.get_node_from_id(id)
                    for batch_idx, (data, target) in enumerate(subset.lst):
                        lucky_node.update_dataset_per_client_size(len(data), self.learning.dataset_len,
                                                                  self.learning.dataset_size)  # do not need target?
                        data = data.send(lucky_node.worker)
                        target = target.send(lucky_node.worker)
                        lucky_node.dataset.append((data, target))
                    lucky_node.datasetid = subset.id
                    lucky_node.is_selected = True
                    self.learning.mark_selected_subdataset(subset.id)
                else:
                    Global.mydebug('getting subset id= None')
                    pass
            #Global.mydebug('AFTERRRRR Round='+str(self.myround))
            #Global.mydebug('Noise factor=' + str(self.learning.noise_repeats))
            #Global.mydebug(str(self.learning.debug_subdatasets()))

            if self._ML=='CL':
                self.run_CL()
            elif self._ML=='FL':
                self.run_FL()
            elif self._ML=='EL':
                self.run_EL()

            self.do_csv_logger()
            gc.collect()
            self.active_learners=[]
            self.myround = self.myround + 1
            Global.mydebug(str(datetime.datetime.now()) + ' (debug) I have used datasets='+str(self.learning.get_list_of_used_datasets()))
            Global.mydebug(str(datetime.datetime.now()) + ' (debug) with num=='+str(len(self.learning.get_list_of_used_datasets())))
        Global.mydebug( str(datetime.datetime.now()) + ' (debug) LETS FINISH!')

    def do_csv_logger(self):
        csv_logger = str(self._ML) + ',' + str(self._MODEL) + ',' + str(self._WIRELESS) + ',' \
                     + str(self._DATE_A) + ',' + str(self._DATE_B) + ',' + str(self._DATE_C) + ',' \
                     + str(self._T_BEGIN) + ',' + str(self._T_END) + ',' \
                     + str(self._IID) + ','+ str(self._EED) + ',' \
                     + str(self._DEVICES_PER_ROUND) + ',' + str(self._SUBDATASETS) + ',' \
                     + str(self._AGGREGATION_FACTOR)+ ',' +str(self._WAITING_FACTOR_CL)+','+ str(self._LOCAL_EPOCHS) + ',' \
                     + str(self._BATCH_SIZE) + ',' + str(self._LEARNING_RATE) + ','


        csv_logger = csv_logger + str(self.CURRENT_TIME)

        loss, acc= self.learning.do_test(self.central_model)
        conf=[0]
        Global.mydebug('(debug) Metric loss =' + str(loss) + ' acc=' + str(acc))
        csv_logger = csv_logger + ',' + str(loss) + ',' + str(acc)

        Global.mydebug('(debug) Clients MB UL =' + str(self.devices.calc_total_mb_UL()))
        csv_logger = csv_logger + ',' + str(self.devices.calc_total_mb_UL())
        Global.mydebug('(debug) Clients MB DL =' + str(self.devices.calc_total_mb_DL()))
        csv_logger = csv_logger + ',' + str(self.devices.calc_total_mb_DL())
        Global.mydebug('(debug) Clients mblost UL =' + str(self.devices.calc_total_mb_lost_UL()))
        csv_logger = csv_logger + ',' + str(self.devices.calc_total_mb_lost_UL())
        Global.mydebug('(debug) Clients mblost DL =' + str(self.devices.calc_total_mb_lost_DL()))
        csv_logger = csv_logger + ',' + str(self.devices.calc_total_mb_lost_DL())

        Global.mydebug('(debug) Clients time UL =' + str(self.devices.calc_total_time_spent_UL()))
        csv_logger = csv_logger + ',' + str(self.devices.calc_total_time_spent_UL())
        Global.mydebug('(debug) Clients time DL =' + str(self.devices.calc_total_time_spent_DL()))
        csv_logger = csv_logger + ',' + str(self.devices.calc_total_time_spent_DL())
        Global.mydebug('(debug) Clients time proc =' + str(self.devices.calc_total_time_spent_proc()))
        csv_logger = csv_logger + ',' + str(self.devices.calc_total_time_spent_proc())

        Global.mydebug('(debug) Clients energy UL _succ =' + str(self.devices.calc_total_energy_spent_UL_succ()))
        csv_logger = csv_logger + ',' + str(self.devices.calc_total_energy_spent_UL_succ())
        Global.mydebug('(debug) Clients energy DL succ=' + str(self.devices.calc_total_energy_spent_DL_succ()))
        csv_logger = csv_logger + ',' + str(self.devices.calc_total_energy_spent_DL_succ())
        Global.mydebug('(debug) Clients energy proc succ=' + str(self.devices.calc_total_energy_spent_proc_succ()))
        csv_logger = csv_logger + ',' + str(self.devices.calc_total_energy_spent_proc_succ())

        Global.mydebug('(debug) Clients energy UL _fail =' + str(self.devices.calc_total_energy_spent_UL_fail()))
        csv_logger = csv_logger + ',' + str(self.devices.calc_total_energy_spent_UL_fail())
        Global.mydebug('(debug) Clients energy DL _fail=' + str(self.devices.calc_total_energy_spent_DL_fail()))
        csv_logger = csv_logger + ',' + str(self.devices.calc_total_energy_spent_DL_fail())
        Global.mydebug('(debug) Clients energy proc _fail=' + str(self.devices.calc_total_energy_spent_proc_fail()))
        csv_logger = csv_logger + ',' + str(self.devices.calc_total_energy_spent_proc_fail())

        Global.mydebug('(debug) Edge energy proc =' + str(self.mecs.calc_total_energy_spent_proc()))
        csv_logger = csv_logger + ',' + str(self.mecs.calc_total_energy_spent_proc())
        Global.mydebug('(debug) Edge time proc =' + str(self.mecs.calc_total_time_spent_proc()))
        csv_logger = csv_logger + ',' + str(self.mecs.calc_total_time_spent_proc())
        Global.mydebug('(debug) Edge _mb_UL =' + str(self.mecs.calc_total_mb_UL()))
        csv_logger = csv_logger + ',' + str(self.mecs.calc_total_mb_UL())
        Global.mydebug('(debug) Edge _mb_DL =' + str(self.mecs.calc_total_mb_DL()))
        csv_logger = csv_logger + ',' + str(self.mecs.calc_total_mb_DL())

        Global.mydebug('(debug) Core energy bs to edge =' + str(self.core.energy_bs_to_edge))
        csv_logger = csv_logger + ',' + str(self.core.energy_bs_to_edge)
        Global.mydebug('(debug) Core energy edge to cloud =' + str(self.core.energy_edge_to_cloud))
        csv_logger = csv_logger + ',' + str(self.core.energy_edge_to_cloud)
        Global.mydebug('(debug) Core time bs to edge =' + str(self.core.time_bs_to_edge))
        csv_logger = csv_logger + ',' + str(self.core.time_bs_to_edge)
        Global.mydebug('(debug) Core time edge to cloud =' + str(self.core.time_edge_to_cloud))
        csv_logger = csv_logger + ',' + str(self.core.time_edge_to_cloud)

        Global.mydebug('(debug) Core energy edge to bs =' + str(self.core.energy_edge_to_bs))
        csv_logger = csv_logger + ',' + str(self.core.energy_edge_to_bs)
        Global.mydebug('(debug) Core energy cloud to edge =' + str(self.core.energy_cloud_to_edge))
        csv_logger = csv_logger + ',' + str(self.core.energy_cloud_to_edge)
        Global.mydebug('(debug) Core time edge to bs =' + str(self.core.time_edge_to_bs))
        csv_logger = csv_logger + ',' + str(self.core.time_edge_to_bs)
        Global.mydebug('(debug) Core time cloud to edge =' + str(self.core.time_cloud_to_edge))
        csv_logger = csv_logger + ',' + str(self.core.energy_cloud_to_edge)

        Global.mydebug('(debug) Cloud time =' + str(self.cloud.time))
        csv_logger = csv_logger + ',' + str(self.cloud.time)
        Global.mydebug('(debug) Cloud energy =' + str(self.cloud.energy))
        csv_logger = csv_logger + ',' + str(self.cloud.energy)

        #confstr = ''
        #for label in conf:
        #    for value in label:
        #        confstr = confstr + str(value) + ';'
        #    confstr = confstr[:-1]
        #    confstr = confstr + '-'

        #csv_logger = csv_logger + ',' + str(confstr)
        csv_logger = csv_logger + ',' + str(self.myround)
        csv_logger = csv_logger + ',' + str(self.active_learners)

        if self._EXPORT_CSV:
            with open(Global._ROOT + Global._LOGS_FOLDER + Global._END_CSV_NAME, mode='a') as file:
                file.write(csv_logger + '\n')

    def run_CL(self):
        max_parallel_time=0
        num_of_active_clients=0
        for node in self.devices.get_selected_list():
            # upload model - parallel
            estimate_ul_time=float(node.dataset_size/self.network.get_speed_ul(self.mobility.get_total_present_now_filtered_by_cell(self.CURRENT_TIME,node.name)))

            # upload validity test
            if (not self.mobility.can_tx_rx(node.name,estimate_ul_time+self.CURRENT_TIME)):
                Global.mydebug('(debug) Fail to upload content for subset =' + str(node.datasetid))
                self.learning.unmark_selected_subdataset(node.datasetid)
                self.devices.set_node_selection(node.id,False)
                self.devices.add_node_mblostUL(node.id,node.dataset_size)
                self.devices.add_node_energyUL_fail(node.id,self.energy_trx.get_spent_ul(estimate_ul_time))
                self.devices.add_node_timeUL(node.id,estimate_ul_time)
                max_parallel_time=max(max_parallel_time,estimate_ul_time)
                continue
            Global.mydebug('(debug) Succ to upload content for subset =' + str(node.datasetid))
            # upload_process - successful use of node
            self.learning.mark_used_subdataset(node.datasetid)
            self.devices.set_node_chosen_before(node.id, True)
            self.devices.add_node_mbUL(node.id, node.dataset_size)
            self.devices.add_node_energyUL_succ(node.id, self.energy_trx.get_spent_ul(estimate_ul_time))
            self.devices.add_node_timeUL(node.id, estimate_ul_time)

            # core upload
            total_mb = node.dataset_size
            self.core.add_energy_UL(total_mb)
            self.core.add_time_UL(total_mb)

            max_parallel_time = max(max_parallel_time, estimate_ul_time +self.core.calc_time_UL(total_mb))
            num_of_active_clients=num_of_active_clients=+1

        self.active_learners.append(num_of_active_clients)
        self.CURRENT_TIME=self.CURRENT_TIME+max_parallel_time
        Global.mydebug('(debug) Data uploaded to cloud, time =' +str(self.CURRENT_TIME))

        # CLOUD - aggreagation check
        if (len(self.devices.get_selected_list())<1):
            for node in self.devices.get_selected_list():
                self.learning.unmark_used_subdataset(node.datasetid)
            self.learning.unmark_all_selected()
            self.devices.clear_all_chosen_before()
            self.devices.clear_all_selected()
            Global.mydebug('(debug) ERROR not enough devices to perform aggregation - myround ended =' )
            self.myround=self.myround+1
        else:
            # aggregation cloud - central training(time,resources=0)
            list_of_datasets = self.devices.get_dataset_list_from_selected_clients()
            print('List of datasets='+str(len(list_of_datasets)))
            merged_dataset = []
            for dataset in list_of_datasets:
                for item in dataset:
                    new_data = item[0].get()
                    new_target = item[1].get()
                    merged_dataset.append((new_data, new_target))
            print('List of merged_dataset='+str(len(merged_dataset)))
            # Random shuffle inter-batch
            # random.shuffle(merged_dataset)

            if self._WAITING_FACTOR_CL==0:
                for epoch in range(0,self._LOCAL_EPOCHS):
                    for batch_idx, (data, target) in enumerate(merged_dataset):
                        self.central_optimizer.zero_grad()
                        output = self.central_model(data)
                        central_loss = F.nll_loss(output, target)
                        central_loss.backward()
                        self.central_optimizer.step()
                samples = len(merged_dataset) * self.learning.batch_size
            else:
                self.merged_dataset.extend(merged_dataset)
                total_len=531131

                current_superdataset_len=len(self.merged_dataset) * self.learning.batch_size
                print('So far superdataset len='+str(current_superdataset_len))
                print('Total len='+str(531131))

                if current_superdataset_len>=total_len*(self._WAITING_FACTOR_CL/100):
                    if self.has_begun_superbatch_training:
                        to_train_dataset=merged_dataset
                    else:
                        to_train_dataset=self.merged_dataset
                        self.has_begun_superbatch_training=True
                    random.shuffle(to_train_dataset)
                    for epoch in range(0, self._LOCAL_EPOCHS):
                        for batch_idx, (data, target) in enumerate(to_train_dataset):
                            self.central_optimizer.zero_grad()
                            output = self.central_model(data)
                            central_loss = F.nll_loss(output, target)
                            central_loss.backward()
                            self.central_optimizer.step()
                    samples = len(to_train_dataset) * self.learning.batch_size
                else:
                    samples=0

            self.cloud.add_energy_CL(samples, self._LOCAL_EPOCHS)
            self.cloud.add_time_CL(samples, self._LOCAL_EPOCHS)
            self.CURRENT_TIME = self.CURRENT_TIME + self.cloud.get_time_train_with_epochs(samples,self._LOCAL_EPOCHS)
            Global.mydebug('(debug) Cloud training finished - time =' + str(self.CURRENT_TIME))
            # Finish myround and print central model eval results
            self.devices.clear_all_selected()
            self.learning.unmark_all_selected()
            Global.mydebug(str(datetime.datetime.now()) + ' (debug2) I have used datasets='+str(self.learning.get_list_of_used_datasets()))
            Global.mydebug(str(datetime.datetime.now()) + ' (debug2) with num=='+str(len(self.learning.get_list_of_used_datasets())))

    def run_FL(self):
        # federated learning per node
        max_parallel_time=0
        for node in self.devices.get_selected_list():

            # broadcast model
            core_time=float(self.core.calc_time_DL(self.central_model.get_size()))
            access_time=float(self.central_model.get_size()/self.network.get_speed_dl(self.mobility.get_total_present_now_filtered_by_cell(self.CURRENT_TIME,node.name)))
            estimate_dl_time=core_time+access_time
            local_node_time=estimate_dl_time

            # download validity check
            if (not self.mobility.can_tx_rx(node.name,self.CURRENT_TIME+local_node_time)):
                Global.mydebug('(debug) Fail to download content for =' + str(node.name))
                self.learning.unmark_selected_subdataset(node.datasetid)
                self.devices.set_node_selection(node.id,False)
                self.devices.add_node_mblostDL(node.id,self.central_model.get_size())
                self.devices.add_node_energyDL_fail(node.id,self.energy_trx.get_spent_dl(local_node_time))
                self.devices.add_node_timeDL(node.id,node.timeDL+access_time)
                self.core.add_energy_DL(self.central_model.get_size())
                self.core.add_time_DL(self.central_model.get_size())
                max_parallel_time=max(max_parallel_time,local_node_time)
                continue

            # download process
            node.model = self.central_model.copy().send(node.worker)
            node.optimizer = optim.SGD(params=node.model.parameters(), lr=self.learning.learning_rate)
            node.params = list(node.model.parameters())
            self.devices.add_node_mbDL(node.id, self.central_model.get_size())
            self.devices.add_node_energyDL_succ(node.id, self.energy_trx.get_spent_dl(local_node_time))
            self.devices.add_node_timeUL(node.id, node.timeDL + access_time)
            self.core.add_energy_DL(self.central_model.get_size())
            self.core.add_time_DL(self.central_model.get_size())

            # local training
            for epoch in range(0,self._LOCAL_EPOCHS):
                for batch_idx, (data, target) in enumerate(node.dataset):
                    if (data.location != node.worker):
                        Global.mydebug('(debug) ERROR - Weird location')
                        continue
                    node.optimizer.zero_grad()
                    output = node.model(data)
                    node.loss = F.nll_loss(output, target)
                    node.loss.backward()
                    node.optimizer.step()

            samples=len(node.dataset)*self.learning.batch_size
            estimate_train_time=self.central_model.get_train_time_mobile_with_epochs(samples,self._LOCAL_EPOCHS)
            local_node_time=local_node_time+estimate_train_time
            self.devices.add_node_energyPROC_temp(node.id,self.energy_proc.get_spent(estimate_train_time))
            self.devices.add_node_timePROC(node.id,estimate_train_time)
            max_parallel_time = max(max_parallel_time, local_node_time)

        self.CURRENT_TIME=self.CURRENT_TIME+max_parallel_time
        Global.mydebug('(debug) FL and process complete, CURRENT_TIME =' + str(self.CURRENT_TIME))

        # upload
        max_parallel_time=0
        num_of_active_clients=0
        for node in self.devices.get_selected_list():
            # upload model - parallel
            estimate_ul_time=float(self.central_model.get_size()/self.network.get_speed_ul(self.mobility.get_total_present_now_filtered_by_cell(self.CURRENT_TIME, node.name)))
            # upload validity test
            if (not self.mobility.can_tx_rx(node.name,estimate_ul_time+self.CURRENT_TIME)):
                Global.mydebug('(debug) Fail to upload content for =' + str(node.name))
                self.learning.unmark_selected_subdataset(node.datasetid)
                self.devices.set_node_selection(node.id,False)
                self.devices.add_node_mblostUL(node.id,self.central_model.get_size())
                self.devices.add_node_energyUL_fail(node.id,self.energy_trx.get_spent_ul(estimate_ul_time))
                self.devices.add_node_energyPROC_fail(node.id)
                self.devices.add_node_timeUL(node.id,estimate_ul_time)
                max_parallel_time=max(max_parallel_time,estimate_ul_time)
                continue

            # upload_process - successful use of node
            self.learning.mark_used_subdataset(node.datasetid)
            self.devices.set_node_chosen_before(node.id, True)
            self.devices.add_node_mbUL(node.id, self.central_model.get_size())
            self.devices.add_node_energyUL_succ(node.id, self.energy_trx.get_spent_ul(estimate_ul_time))
            self.devices.add_node_energyPROC_succ(node.id)
            self.devices.add_node_timeUL(node.id, estimate_ul_time)

            # core upload
            total_mb = self.central_model.get_size()
            self.core.add_energy_UL(total_mb)
            self.core.add_time_UL(total_mb)
            max_parallel_time = max(max_parallel_time, estimate_ul_time +self.core.calc_time_UL(total_mb))
            num_of_active_clients=num_of_active_clients+1

        self.active_learners.append(num_of_active_clients)
        self.CURRENT_TIME=self.CURRENT_TIME+max_parallel_time
        Global.mydebug('(debug) Data uploaded to cloud, time =' +str(self.CURRENT_TIME))

        # aggreagation check
        if len(self.devices.get_selected_list())==0:
            selected_worker_list = None
            Global.mydebug('(debug) No clients to aggregate')
        elif (len(self.devices.get_selected_list())==1):
            sole_list=self.devices.get_selected_list()
            sole_node=sole_list[0]
            phantom_worker = sy.VirtualWorker(self.hook, id='phantom' + str(self.phantom_unique_id))
            self.phantom_unique_id=self.phantom_unique_id+1
            phantom_worker_model = sole_node.model.copy().send(phantom_worker)
            selected_worker_list=self.devices.get_selected_list_workers()
            selected_worker_list.append(phantom_worker)
        else:
            selected_worker_list = self.devices.get_selected_list_workers()

        if selected_worker_list is not None and len(selected_worker_list)>0:
            new_params = list()
            for param_i in range(len(list(self.central_model.parameters()))):
                spdz_params = list()
                for node in self.devices.get_selected_list():
                    spdz_params.append(node.params[param_i].fix_precision().share(*selected_worker_list,crypto_provider=self.crypto_provider).get())
                new_param_sum=0
                for i in spdz_params:
                    new_param_sum=new_param_sum+i
                new_param=new_param_sum.get().float_precision()/len(spdz_params)
                new_params.append(new_param)
            # cleanup
            with torch.no_grad():
                # clear old central model
                for model in list(self.central_model.parameters()):
                    for param in model:
                        param *= 0
                # update central model
                for param_index in range(len(list(self.central_model.parameters()))):
                    list(self.central_model.parameters())[param_index].set_(new_params[param_index])

        aggr_time=self.cloud.get_time_aggr(len(self.devices.get_selected_list_workers()))
        self.CURRENT_TIME=self.CURRENT_TIME+aggr_time
        self.cloud.add_energy_FL(len(self.devices.get_selected_list_workers()))
        self.cloud.add_time_FL(len(self.devices.get_selected_list_workers()))
        Global.mydebug('(debug) Cloud aggrp finito - time =' +str(self.CURRENT_TIME))

        # Finish myround and print central model eval results
        self.clean()
        self.learning.unmark_all_selected()

    def run_EL(self):
        # upload data to BSs
        max_parallel_time=0
        for node in self.devices.get_selected_list():
            # upload model - parallel
            estimate_ul_time=float(node.dataset_size/self.network.get_speed_ul(self.mobility.get_total_present_now_filtered_by_cell(self.CURRENT_TIME,node.name)))
            # upload validity test
            if (not self.mobility.can_tx_rx(node.name,estimate_ul_time+self.CURRENT_TIME)):
                Global.mydebug('(debug) Fail to upload content for =' + str(node.id))
                node.is_selected=False
                self.learning.unmark_selected_subdataset(node.datasetid)
                node.mblostUL=node.mblostUL+node.dataset_size
                node.energyUL=node.energyUL+self.energy_trx.get_spent_ul(estimate_ul_time)
                node.timeUL=node.timeUL+estimate_ul_time
                max_parallel_time=max(max_parallel_time,estimate_ul_time)
                continue

            # upload_process - successful use of node
            max_parallel_time = max(max_parallel_time, estimate_ul_time)
            self.learning.mark_used_subdataset(node.datasetid)
            node.mec_id=self.mobility.get_mec_from_clientid_time(self.CURRENT_TIME+estimate_ul_time,node.name)
            node.was_chosen_before=True
            node.mbUL=node.mbUL+node.dataset_size
            node.energyUL = node.energyUL + self.energy_trx.get_spent_ul(estimate_ul_time)
            node.timeUL = node.timeUL + estimate_ul_time
            self.mecs.set_active(node.mec_id)

        self.CURRENT_TIME=self.CURRENT_TIME+max_parallel_time
        Global.mydebug('(debug) Access UL finito - time =' +str(self.CURRENT_TIME))

        # UL BS to edge - todo can be done in parallel manner per BS
        total_mb=0
        for node in self.devices.get_selected_list():
            total_mb=total_mb+node.dataset_size

        self.CURRENT_TIME=self.CURRENT_TIME+self.core.calc_time_bs_to_edge(total_mb)
        self.core.add_energy_bs_to_edge(total_mb)
        self.core.add_time_bs_to_edge(total_mb)
        Global.mydebug('(debug) BS to edge upload finish =' +str(self.CURRENT_TIME))

        # broadcast model to active mecs
        max_parallel_time = 0
        for mec in self.mecs.get_active_mecs():
            local_mec_time = float(self.core.calc_time_cloud_to_edge(self.central_model.get_size()))
            # download process
            mec.model = self.central_model.copy().send(mec.worker)
            mec.optimizer = optim.SGD(params=mec.model.parameters(), lr=self.learning.learning_rate)
            mec.params = list(mec.model.parameters())
            mec.mbDL = mec.mbDL + self.central_model.get_size()
            self.core.add_energy_cloud_to_edge(self.central_model.get_size())
            self.core.add_time_cloud_to_edge(self.central_model.get_size())
            max_parallel_time = max(max_parallel_time, local_mec_time)
        self.CURRENT_TIME = self.CURRENT_TIME + max_parallel_time

        # Edge - training
        max_parallel_time = 0
        for mec in self.mecs.get_active_mecs():
            list_of_datasets = self.devices.get_dataset_list_from_selected_clients_per_mec(mec.id)
            #Global.mydebug('(debug) Datasets in this mec='+str(len(list_of_datasets)))
            mec.dataset = []

            for dataset in list_of_datasets:
                for item in dataset:
                    new_data = item[0].get()
                    new_target = item[1].get()
                    data = new_data.send(mec.worker)
                    target = new_target.send(mec.worker)
                    mec.dataset.append((data,target))
                    mec.update_dataset_per_client_size(len(new_data), self.learning.dataset_len,self.learning.dataset_size)

            for batch_idx, (data, target) in enumerate(mec.dataset):
                if (data.location != mec.worker):
                    Global.mydebug('(debug) ERROR - Weird location')
                    continue
                mec.optimizer.zero_grad()
                output = mec.model(data)
                mec.loss=F.nll_loss(output, target)
                mec.loss.backward()
                mec.optimizer.step()

            mec.calc_metrics_training(self.learning.batch_size)
            self.active_learners.append(len(self.devices.get_selected_list_per_mec(mec.id)))
            max_parallel_time=max(max_parallel_time,mec.calc_time_training(self.learning.batch_size))

        self.CURRENT_TIME = self.CURRENT_TIME + max_parallel_time
        Global.mydebug('(debug) Edge training finito - time =' + str(self.CURRENT_TIME))
        self.devices.clean()
        self.mecs.clean_inactive()

        # edge to cloud
        max_parallel_time = 0
        for mec in self.mecs.get_active_mecs():
            local_mec_time = float(self.core.calc_time_edge_to_cloud(self.central_model.get_size()))
            # upload process
            mec.mbUL = mec.mbUL + self.central_model.get_size()
            self.core.add_energy_edge_to_cloud(self.central_model.get_size())
            self.core.add_time_edge_to_cloud(self.central_model.get_size())
            max_parallel_time = max(max_parallel_time, local_mec_time)
        self.CURRENT_TIME = self.CURRENT_TIME + max_parallel_time
        Global.mydebug('(debug) Models uploaded to cloud =' +str(self.CURRENT_TIME))

        # aggreagation check
        if len(self.mecs.get_active_mecs())==0:
            active_mec_workers = []
            Global.mydebug('(debug) No clients to aggregate')
        elif (len(self.mecs.get_active_mecs())==1):
            sole_list=self.mecs.get_active_mecs()
            sole_node=sole_list[0]
            phantom_worker = sy.VirtualWorker(self.hook, id='phantom' + str(self.phantom_unique_id))
            self.phantom_unique_id=self.phantom_unique_id+1
            phantom_worker_model = sole_node.model.copy().send(phantom_worker)
            active_mec_workers=self.mecs.get_active_mecs_workers()
            active_mec_workers.append(phantom_worker)
        else:
            active_mec_workers=self.mecs.get_active_mecs_workers()
        Global.mydebug('Active mec workers='+str(len(active_mec_workers)))
        if len(active_mec_workers)>0:
            new_params = list()
            for param_i in range(len(list(self.central_model.parameters()))):
                spdz_params = list()
                for mec in self.mecs.get_active_mecs():
                    spdz_params.append(mec.params[param_i].fix_precision().share(*active_mec_workers,crypto_provider=self.crypto_provider).get())
                new_param_sum=0
                for i in spdz_params:
                    new_param_sum=new_param_sum+i
                new_param=new_param_sum.get().float_precision()/len(spdz_params)
                new_params.append(new_param)
            # cleanup
            with torch.no_grad():
                # clear old central model
                for model in list(self.central_model.parameters()):
                    for param in model:
                        param *= 0
                # update central model
                for param_index in range(len(list(self.central_model.parameters()))):
                    list(self.central_model.parameters())[param_index].set_(new_params[param_index])

        aggr_time=self.cloud.get_time_aggr(len(self.mecs.get_active_mecs_workers()))
        self.CURRENT_TIME=self.CURRENT_TIME+aggr_time
        self.cloud.add_energy_FL(len(self.mecs.get_active_mecs_workers()))
        self.cloud.add_time_FL(len(self.mecs.get_active_mecs_workers()))
        Global.mydebug('(debug) Cloud aggrp finito - time =' +str(self.CURRENT_TIME))

        # Finish myround and print central model eval results
        self.clean()
        self.learning.unmark_all_selected()