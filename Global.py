# Root
_ROOT='C:\Pycharm\Projects\\fl_tests\\'
_MODELS_FOLDER='outcome_models\\'
_DATASETS_FOLDER='training_datasets\\'
_MOBILITY_FOLDER='mobility_datasets\\'
_LOGS_FOLDER='logs\\'
_END_CSV_NAME='end_csv.txt'
_QMNIST_DATASET_PATH='QMNIST\\processed\\train.pt'
_INFIMNIST_DATASET_PATH='MNIST\\processed\\training.pt'
_CIFAR10_DATASET_PATH='cifar-10-batches-py\\'
_SVHN_PATH='SVHN\\'
_SVHN_DATASET_PATH='SVHN\\extra_32x32.mat'
_SHANGHAI_1='Shanghai_Sheet1.csv'
_SHANGHAI_2='Shanghai_sheet2.csv'
_SHANGHAI_1_DAY_1='Shanghai_Sheet1_day1.csv'
#_WIFIDOG='5g_20200803.csv'
_WIFIDOG='wifidog_exported.csv'
_DEBUG_FILENAME=None

# Client settings - WifiDog traffic
#YEAR=2010
#MONTH=3
#DAY=8

# Client settings - Shanghai
#SHANGHAI_DATE='21/6/2014'
#SHANGHAI_BS='31.253346/121.448039'

# ML
_MODEL_CLASSES=62
BATCH_SIZE=64
T_BEGIN=3600*1
T_END=3600*2
MEAN_UL=0.2
SD_UL=0.05
MEAN_DL=0.5
SD_DL=0.15
MIN_UL_DL=0.1
Z=0.1

# FL
MEAN_PROC_TIME=2
SD_PROC_TIME=0.2
MIN_PROC_TIME=0.1


def mydebug(mystr):
    with open(_ROOT + _LOGS_FOLDER + _DEBUG_FILENAME + ".txt", mode='a') as file:
        file.write(mystr + '\n')