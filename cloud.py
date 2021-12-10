# Methodology to Predict the Power Consumption of Servers in Data Centres
# mobile train 125 (55 or 350) samples per sec -resnet
# https://arxiv.org/pdf/1906.04278.pdf

# cloud train 0.1 msec per sample (https://arxiv.org/ftp/arxiv/papers/1812/1812.11731.pdf)
# cloud agrr=0.4 secs per model (linear, since mine agrr=1.56, mine train=6250 per sec)
TRAIN_SPEED=40000# samples/sec
class Cloud:
    def __init__(self):
        self.power_train=float(125) #watt for 90 perc cpu
        self.power_aggr=float(15) #watt for 10 perc cpu
        self.energy=0
        self.time=0

    def get_time_train_per_epoch(self,samples):
        return (samples/TRAIN_SPEED) #sec

    def get_time_train_with_epochs(self,samples,epochs):
        return epochs*self.get_time_train_per_epoch(samples)

    def get_energy_train_per_epoch(self,samples):
        return self.get_time_train_per_epoch(samples)*self.power_train

    def get_energy_train_with_epochs(self,samples,epochs):
        return epochs*self.get_energy_train_per_epoch(samples)

    def get_time_aggr(self,modelcount):
        return 0.4*modelcount #sec

    def get_energy_aggr(self,modelcount):
        print(str(modelcount))
        return self.get_time_aggr(modelcount)*self.power_aggr

    def add_energy_CL(self,samples,epochs):
        self.energy = self.energy + self.get_energy_train_with_epochs(samples,epochs)

    def add_energy_FL(self,modelcount):
        self.energy = self.energy + self.get_energy_aggr(modelcount)

    def add_time_CL(self,samples,epochs):
        self.time = self.time + self.get_time_train_with_epochs(samples,epochs)

    def add_time_FL(self,modelcount):
        self.time = self.time + self.get_time_aggr(modelcount)