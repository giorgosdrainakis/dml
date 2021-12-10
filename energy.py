POWER_TRAIN_CPU_INCEPTION=float(2) #watt
POWER_TRAIN_CPU_DRL=float(1.8) #watt
POWER_WIFI_UL=float(0.75) # watt
POWER_WIFI_DL=float(0.25) # watt
POWER_LTE_UL=float(2.2) # watt
POWER_LTE_DL=float(1.5) # watt

class Energy_Wifi:  # E=Work*time (average values)
    def get_spent_ul(self,time):
        return float(POWER_WIFI_UL*time)

    def get_spent_dl(self,time):
        return float(POWER_WIFI_DL*time)

class Energy_LTE:
    def get_spent_ul(self,time):
        return float(POWER_LTE_UL*time)

    def get_spent_dl(self,time):
        return float(POWER_LTE_DL*time)

class Energy_CPU_Inception:
    def get_spent(self,time):
        return float(POWER_TRAIN_CPU_INCEPTION*time)

class Energy_CPU_DRL:
    def get_spent(self,time):
        return float(POWER_TRAIN_CPU_DRL*time)

