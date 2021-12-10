import random
import Tools

# params
# LTE
#https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6834753
LTE_DL_CELL_AVG=53.8 # Mbps
LTE_DL_SD_FACTOR=0.2
LTE_DL_MIN_USER=1.8
LTE_UL_CELL_AVG=47.2 # Mbps
LTE_UL_SD_FACTOR=0.2
LTE_UL_MIN_USER=1.94
# WIFI - 802.11g
#Achilles and the Tortoise: Power Consumption in IEEE 802.11n and IEEE 802.11g Networks
#Uplink and Downlink Coverage Improvements of 802.11g Signals Using a Distributed Antenna Network
WIFI_DL_CELL_AVG=19 # Mbps
WIFI_DL_SD_FACTOR=0.2
WIFI_DL_MIN_USER=0.24
WIFI_UL_CELL_AVG=18 # Mbps
WIFI_UL_SD_FACTOR=0.2
WIFI_UL_MIN_USER=0.24

class Access_LTE():
    def __init__(self):
        self.dl_cell_avg=LTE_DL_CELL_AVG
        self.dl_sd=LTE_DL_SD_FACTOR
        self.dl_min=LTE_DL_MIN_USER
        self.ul_cell_avg=LTE_UL_CELL_AVG
        self.ul_sd=LTE_UL_SD_FACTOR
        self.ul_min=LTE_UL_MIN_USER

    def get_speed_dl(self,online_users):
        min_speed=self.dl_min
        if (online_users==0):
            avg_user_speed = self.dl_cell_avg
        else:
            avg_user_speed=self.dl_cell_avg/online_users
        new_random_speed=random.gauss(avg_user_speed, avg_user_speed*self.dl_sd)
        return Tools.mbps_to_mb_per_sec(max(min_speed,new_random_speed))

    def get_speed_ul(self,online_users):
        min_speed=self.ul_min
        if (online_users==0):
            avg_user_speed = self.ul_cell_avg
        else:
            avg_user_speed=self.ul_cell_avg/online_users
        new_random_speed=random.gauss(avg_user_speed, avg_user_speed*self.ul_sd)
        return Tools.mbps_to_mb_per_sec(max(min_speed,new_random_speed))

class Access_WIFI():
    def __init__(self):
        self.dl_cell_avg=WIFI_DL_CELL_AVG
        self.dl_sd=WIFI_DL_SD_FACTOR
        self.dl_min=WIFI_DL_MIN_USER
        self.ul_cell_avg=WIFI_UL_CELL_AVG
        self.ul_sd=WIFI_UL_SD_FACTOR
        self.ul_min=WIFI_UL_MIN_USER

    def get_speed_dl(self,online_users):
        min_speed=self.dl_min
        if (online_users==0):
            avg_user_speed = self.dl_cell_avg
        else:
            avg_user_speed=self.dl_cell_avg/online_users
        new_random_speed=random.gauss(avg_user_speed, avg_user_speed*self.dl_sd)
        return Tools.mbps_to_mb_per_sec(max(min_speed,new_random_speed))

    def get_speed_ul(self,online_users):
        min_speed=self.ul_min
        if (online_users==0):
            avg_user_speed = self.ul_cell_avg
        else:
            avg_user_speed=self.ul_cell_avg/online_users
        new_random_speed=random.gauss(avg_user_speed, avg_user_speed*self.ul_sd)
        return Tools.mbps_to_mb_per_sec(max(min_speed,new_random_speed))