import sys
from experiment import Experiment

def main(_DATE_A,_DATE_B,_DATE_C,_T_BEGIN,_T_END,_SUBDATASETS,_DEVICES_PER_ROUND,
         _IID,_EED,_WIRELESS,_MODEL,_EXPORT_CSV,_USE_LOGGER,
         _ML,_AGGREGATION_FACTOR,_WAITING_FACTOR_CL,_LOCAL_EPOCHS,_BATCH_SIZE,_LEARNING_RATE):

    exp=Experiment(_DATE_A, _DATE_B, _DATE_C, _T_BEGIN, _T_END, _SUBDATASETS, _DEVICES_PER_ROUND,
                  _IID,_EED, _WIRELESS, _MODEL, _EXPORT_CSV, _USE_LOGGER,
                   _ML, _AGGREGATION_FACTOR,_WAITING_FACTOR_CL,_LOCAL_EPOCHS,_BATCH_SIZE,_LEARNING_RATE)

if __name__=='__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],
         sys.argv[7],sys.argv[8],sys.argv[9],sys.argv[10],sys.argv[11],sys.argv[12],
         sys.argv[13],sys.argv[14],sys.argv[15],sys.argv[16],sys.argv[17],sys.argv[18],sys.argv[19])