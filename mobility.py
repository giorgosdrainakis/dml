import csv
import math
import random
from collections import Counter

from sklearn.cluster import KMeans

import Global

ONLINE_TIME_THRESHOLD=90 #sec
ONLINE_TIME_THRESHOLD_WIFI=0
#WIFIDOG_WANTED_DATE=[2010,3,8]

class Record_Shanghai():
    def __init__(self,start_time,end_time,location,id):
        self.location=location
        self.id=id
        self.start_date=None
        self.sec_in=None
        self.end_date=None
        self.sec_out=None
        self.is_pulled=False
        self.locationX=None
        self.locationY=None
        self.mec_id=None

        if ' ' in start_time:
            self.start_date,real_start_time=start_time.split(' ',1)
            if ':' in real_start_time:
                real_start_hour,real_start_mins=real_start_time.split(':',1)
                self.sec_in= int(real_start_hour)*3600 + int(real_start_mins)*60  # in secs

        if ' ' in end_time:
            self.end_date, real_end_time = end_time.split(' ', 1)
            if ':' in real_end_time:
                real_end_hour, real_end_mins = real_end_time.split(':', 1)
                self.sec_out = int(real_end_hour) * 3600 + int(real_end_mins) * 60  # in secs

    def show(self):
        print('ID:'+str(self.id)+',DateIN:'+str(self.start_date)+',TimeIN:'+str(self.sec_in)+',DateOUT:'+str(self.end_date)+',TimeOUT:'+str(self.sec_out)+',BS:'+str(self.location))

class Record_Wifidog():
    def __init__(self,user_id,stamp_in,stamp_out,location):
        self.location=location
        self.id = user_id
        self.year_in = int(stamp_in[0:4])
        self.month_in = int(stamp_in[5:7])
        self.day_in=int(stamp_in[8:10])
        self.sec_in=int(stamp_in[11:13]) * 3600 + int(stamp_in[14:16]) * 60 + int(stamp_in[17:19])
        self.year_out = int(stamp_out[0:4])
        self.month_out = int(stamp_out[5:7])
        self.day_out=int(stamp_out[8:10])
        self.sec_out=int(stamp_out[11:13]) * 3600 + int(stamp_out[14:16]) * 60 + int(stamp_out[17:19])
        self.is_pulled=False

    def show(self):
        print('ID:'+str(self.id)+',TimeIN:'+str(self.sec_in)+',TimeOUT:'+str(self.sec_out)+',BS:'+str(self.location))

class Mobility_Wifi():
    def __init__(self,WIFIDOG_WANTED_DATE,users_pull,t_begin,t_end):
        self.db=[]
        self.t_begin=t_begin
        self.t_end=t_end
        self.wait_for_clients_time=1 #secs
        with open(Global._ROOT+Global._MOBILITY_FOLDER+Global._WIFIDOG) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for row in csv_reader:
                new_record = Record_Wifidog(row['user_id'], row['timestamp_in'], row['timestamp_out'],row['node_id'])
                self.add_record_with_date_and_location_check(WIFIDOG_WANTED_DATE[0],WIFIDOG_WANTED_DATE[1],WIFIDOG_WANTED_DATE[2],new_record)
        self.do_random_user_pull(users_pull)

    def add_record_with_date_and_location_check(self,year,month,day,record):
        is_same_year=(record.year_in==year) and (record.year_out==year)
        is_same_month=(record.month_in==month) and (record.month_out==month)
        is_same_day=(record.day_in==day) and (record.day_out==day)
        location_test_ok=(record.location is not None) and (record.location!='')
        if is_same_day and is_same_month and is_same_year and location_test_ok:
            self.db.append(record)

    def do_random_user_pull(self,users_pull):
        if users_pull==math.inf:
            for record in self.db:
                record.is_pulled=True
        else:
            random_list=random.sample(self.get_present_ids_in_period(self.t_begin, self.t_end), users_pull)
            for record in self.db:
                if record.id in random_list:
                    record.is_pulled=True


    def get_present_ids_in_period(self,tbegin,tend):
        output_list = set()
        for record in self.db:
            if (tbegin <= record.sec_in and record.sec_in <= tend) or (record.sec_in <= tbegin and tbegin <= record.sec_out):
                output_list.add(record.id)
        return list(output_list)

    def get_present_ids_in_period_filtered_by_pull(self,tbegin,tend):
        output_list = set()
        for record in self.db:
            if record.is_pulled:
                if (tbegin <= record.sec_in and record.sec_in <= tend) or (record.sec_in <= tbegin and tbegin <= record.sec_out):
                    output_list.add(record.id)
        return list(output_list)

    def get_present_ids_now_filtered_by_pull(self,time):
        output_list = set()
        for rec in self.db:
            if rec.is_pulled and self.is_id_connected_now(rec.id,time):
                output_list.add(rec.id)
        return list(output_list)

    def get_cell_from_clientid_time(self,time,id):
        for rec in self.db:
            if rec.id==id and rec.sec_in<=time and time<=rec.sec_out:
                return rec.location

    def get_total_present_now_filtered_by_cell(self,time,id):
        mycell=self.get_cell_from_clientid_time(time,id)
        output_list = set()
        for rec in self.db:
            if rec.location==mycell and self.is_id_connected_now(rec.id,time):
                output_list.add(rec.id)
        return len(output_list)

    def can_tx_rx(self, id, future_time):
        if self.is_id_connected_now(id, future_time) or self.is_id_in_ho(id, future_time, ONLINE_TIME_THRESHOLD_WIFI):
            return True
        return False

    def get_id_connection_status_now(self, id, time):
        pass

    def is_id_connected_now(self, id, time):
        for record in self.db:
            if (record.id == id):
                if (record.sec_in <= time) and (time <= record.sec_out):
                    return True
        return False

    def is_id_in_ho(self, id, time, threshold):
        # check if client is during handoff
        if (not self.is_id_connected_now(id, time)) and self.is_id_connected_now(id,time - threshold) and self.is_id_connected_now(id, time + threshold):
            return True
        return False

    def is_id_away(self, id, time):
        pass

class Mobility_Shanghai():
    def __init__(self,users_pull,t_begin,t_end,wanted_date,mec_list):
        self.db=[]
        self.t_begin=t_begin
        self.t_end=t_end
        self.wait_for_clients_time=60 #secs
        self.wanted_date=str(wanted_date[2])+'/'+str(wanted_date[1])+'/'+str(wanted_date[0])
        with open(Global._ROOT+Global._MOBILITY_FOLDER+Global._SHANGHAI_1) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for row in csv_reader:
                new_record = Record_Shanghai(row['start_time'], row['end_time'], row['location'],row['user_id'])
                self.add_record_with_date_and_location_check(new_record,self.wanted_date)

        self.perform_clustering_balanced(mec_list)
        self.do_random_user_pull(users_pull)

    def add_record_with_date_and_location_check(self,record,wanted_date):
        is_same_date=record.start_date==wanted_date and record.end_date==wanted_date
        date_test_ok=(record.sec_in is not None) and (record.sec_out is not None)
        location_test_ok=(record.location is not None) and (record.location!='')
        if is_same_date and date_test_ok and location_test_ok:
            record.locationX, record.locationY = record.location.split('/')
            record.locationX = float(record.locationX)
            record.locationY = float(record.locationY)
            self.db.append(record)

    def do_random_user_pull(self,users_pull):
        if users_pull==math.inf:
            for record in self.db:
                record.is_pulled=True
        else:
            random_list=random.sample(self.get_present_ids_in_period(self.t_begin, self.t_end), users_pull)
            for record in self.db:
                if record.id in random_list:
                    record.is_pulled=True

    def search_mec_by_location(self,location):
        found=None
        for rec in self.db:
            if rec.location==location and rec.mec_id is not None:
                found=rec.mec_id
                break
        return found

    def perform_clustering_balanced(self,mec_list):
        next_id=0
        for rec in self.db:
            potential_mec=self.search_mec_by_location(rec.location)
            if potential_mec is not None:
                rec.mec_id=potential_mec
            else:
                rec.mec_id=mec_list[next_id]
                next_id=next_id+1
                if next_id==len(mec_list):
                    next_id=0

    def perform_clustering(self,mec_list):
        data = []
        for rec in self.db:
            data.append((rec.locationX, rec.locationY))
        kmeans = KMeans(n_clusters=len(mec_list)).fit(data)

        for rec in self.db:
            current_location=[]
            current_location.append((rec.locationX,rec.locationY))
            cluster_id=kmeans.predict(current_location)[0]
            rec.mec_id=mec_list[cluster_id]

    def show(self):
        for record in self.db:
            record.show()

    def get_present_ids_in_period(self,tbegin,tend):
        output_list = set()
        for record in self.db:
            if (tbegin <= record.sec_in and record.sec_in <= tend) or (record.sec_in <= tbegin and tbegin <= record.sec_out):
                output_list.add(record.id)
        return list(output_list)

    def get_present_ids_in_period_filtered_by_pull(self,tbegin,tend):
        output_list = set()
        for record in self.db:
            if record.is_pulled:
                if (tbegin <= record.sec_in and record.sec_in <= tend) or (record.sec_in <= tbegin and tbegin <= record.sec_out):
                    output_list.add(record.id)
        return list(output_list)

    def get_present_ids_now_filtered_by_pull(self,time):
        output_list = set()
        for rec in self.db:
            if rec.is_pulled and self.is_id_connected_now(rec.id,time):
                output_list.add(rec.id)
        return list(output_list)

    def get_cell_from_clientid_time(self,time,id):
        for rec in self.db:
            if rec.id==id and rec.sec_in<=time and time<=rec.sec_out:
                return rec.location

    def get_mec_from_clientid_time(self,time,id):
        for rec in self.db:
            if rec.id==id and rec.sec_in<=time and time<=rec.sec_out:
                return rec.mec_id

    def get_total_present_now_filtered_by_cell(self,time,id):
        mycell=self.get_cell_from_clientid_time(time,id)
        output_list = set()
        for rec in self.db:
            if rec.location==mycell and self.is_id_connected_now(rec.id,time):
                output_list.add(rec.id)
        return len(output_list)

    def can_tx_rx(self,id,future_time):
        if self.is_id_connected_now(id,future_time) or self.is_id_in_ho(id,future_time,ONLINE_TIME_THRESHOLD):
            return True
        return False

    def get_id_connection_status_now(self,id,time):
        pass

    def is_id_connected_now(self,id,time):
        for record in self.db:
            if (record.id==id):
                if (record.sec_in<=time) and (time<=record.sec_out):
                    return True
        return False
        
    def is_id_in_ho(self,id,time,threshold):
        # check if client is during handoff
        if (not self.is_id_connected_now(id,time)) and self.is_id_connected_now(id,time-threshold) and self.is_id_connected_now(id,time+threshold):
            return True
        return False

    def is_id_away(self,id,time):
        pass



def Test_Shanghai_numbers():
    mylist = [400,200, 150, 120, 100, 80, 70, 60, 50, 40, 35, 30, 25, 20]
    for t1 in range(0,3600*24,3600*2):
        for pull in mylist:
            t_begin=t1
            t_end=t1+3600*2
            mobility = Mobility_Shanghai(pull, t_begin, t_end)
            clnum=len(mobility.get_present_ids_in_period(t_begin,t_end))
            clnum2=len(mobility.get_present_ids_in_period_filtered_by_pull(t_begin,t_end))
            print('Timerange:'+str(t_begin)+'-'+str(t_end)+',pull='+str(pull)+',Total clients='+str(clnum)+',pull clients='+str(clnum2))

def Test_Shanghai_HO():
    t1=3600*2
    t2=3600*4
    mobility=Mobility_Shanghai(200,t1,t2)
    ids=mobility.get_present_ids_in_period_filtered_by_pull(t1,t2)
    print('Checking a total clients of ='+str(ids))
    for client_id in ids:
        print('*Checking client='+str(client_id))
        for tt in range(t1,t2):
            if mobility.is_id_in_ho(client_id,tt,ONLINE_TIME_THRESHOLD):
                print('**** FOUND -> '+str(client_id)+' - '+ str(tt))

def Test_Shanghai_tx_rx():
    t1=3600*2
    t2=3600*4
    mobility=Mobility_Shanghai(50,t1,t2)
    ids=mobility.get_present_ids_in_period_filtered_by_pull(t1,t2)
    print('Checking a total clients of ='+str(ids))
    for client_id in ids:
        print('*Client='+str(client_id)+' can trx:'+str(mobility.can_tx_rx(client_id,(t2-t1)/2)))

def Test_wifi_numbers():

    dates=[[2010, 1, 19] ,[2010, 1, 20],[2010, 1, 21], [2010, 1, 25], [2010, 1, 26], \
          [2010, 1, 27],[2010, 1, 28], [2010, 1, 29], [2010, 2, 1], [2010, 2, 2], [2010, 2, 4], [2010, 2, 5]]

    mylist = [400,200, 150, 120, 100, 80, 70, 60, 50, 40, 35, 30, 25, 20]
    for date in dates:
        for pull in mylist:
            t_begin=3600*14
            t_end=3600*16
            mobility = Mobility_Wifi(date,pull, t_begin, t_end)
            clnum=len(mobility.get_present_ids_in_period(t_begin,t_end))
            clnum2=len(mobility.get_present_ids_in_period_filtered_by_pull(t_begin,t_end))
            print('Timerange:'+str(t_begin)+'-'+str(t_end)+',pull='+str(pull)+',Total clients='+str(clnum)+',pull clients='+str(clnum2))

def Test_wifi_HO():
    t1=3600*14
    t2=3600*16
    WIFIDOG_WANTED_DATE = [2010, 3, 8]
    mobility=Mobility_Wifi(WIFIDOG_WANTED_DATE,200,t1,t2)
    ids=mobility.get_present_ids_in_period_filtered_by_pull(t1,t2)
    print('Checking a total clients of ='+str(ids))
    for client_id in ids:
        print('*Checking client='+str(client_id))
        for tt in range(t1,t2):
            if mobility.is_id_in_ho(client_id,tt,ONLINE_TIME_THRESHOLD):
                print('**** FOUND -> '+str(client_id)+' - '+ str(tt))
def Test_wifi_tx_rx():
    t1=3600*14
    t2=3600*16
    WIFIDOG_WANTED_DATE = [2010, 3, 8]
    mobility=Mobility_Wifi(WIFIDOG_WANTED_DATE,50,t1,t2)
    ids=mobility.get_present_ids_in_period_filtered_by_pull(t1,t2)
    print('Checking a total clients of ='+str(ids))
    for client_id in ids:
        print('*Client='+str(client_id)+' can trx:'+str(mobility.can_tx_rx(client_id,t1+2)))

def Test_Shanghai_new():
    mec_list=[1,2,3]
    mobility = Mobility_Shanghai(math.inf, 0, 3600*24, (2014,6,1), mec_list)
    print(len(mobility.db))

