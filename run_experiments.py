import time
import subprocess

def proc_new(total_string_to_send):
    subprocess.run(run_text_total + total_string_to_send)
    print('Finished script for ' + total_string_to_send )

def run():
    #Z_list = [400,360,330,300,260,230,190,150,120,90,60,30,15]
    Z_list =[100]

    #dates_wifi=[(2010,1,19),(2010,1,20),(2010,1,21),(2010,1,25),(2010,1,26),
    #            (2010,1,27),(2010,1,28),(2010,1,29),(2010,2,1),(2010,2,2)]

    # lte
    dates_lte=[(2014,6,1),(2014,6,2),(2014,6,3),(2014,6,4),(2014,6,5),(2014,6,6),(2014,6,7),(2014,6,8),(2014,6,9),(2014,6,10)]

    dates=[(2014,6,4),(2014,6,5)]
    ml_list=['CL']
    _WIRELESS = 'LTE'

    Kappa_list=[10]
    eed_list=[1.7,2,2.3,1000]
    iid_list=[7,5,3,-1]
    waiting_factor_list=[0]

    batch_list =[128]
    local_epoch_list=[25]
    learn_rate_list=[0.1]
    tbegin_list=[0*3600]
    agg_list=[1]

    _MODEL = 'SVHN'
    _EXPORT_CSV = 'True'
    _USE_LOGGER = 'True'

    for _ML in ml_list:
        for _DATE_A,_DATE_B,_DATE_C in dates:
            for _T_BEGIN in tbegin_list:
                _T_END=_T_BEGIN+24*3600
                for _SUBDATASETS in Z_list:
                    for _DEVICES_PER_ROUND in Kappa_list:
                        for _EED in eed_list:
                            for _IID in iid_list:
                                for _LOCAL_EPOCHS in local_epoch_list:
                                    for _BATCH_SIZE in batch_list:
                                        for _LEARNING_RATE in learn_rate_list:
                                            if _ML=='EL':
                                                _WAITING_FACTOR_CL=0
                                                for _AGGREGATION_FACTOR in agg_list:
                                                    total_string_to_send = ' ' + str(_DATE_A) + ' ' + str(_DATE_B) + ' ' + \
                                                        str(_DATE_C) + ' ' + str(_T_BEGIN) + ' ' + str(_T_END) + ' ' + \
                                                        str(_SUBDATASETS) + ' ' + str(_DEVICES_PER_ROUND) + ' ' + str(_IID) + ' '+str(_EED)+' ' + \
                                                        str(_WIRELESS) + ' ' + str(_MODEL) + ' ' + str(_EXPORT_CSV) + ' ' + \
                                                        str(_USE_LOGGER) + ' ' + str(_ML) + ' ' + str(_AGGREGATION_FACTOR)  + ' ' + \
                                                        str(_WAITING_FACTOR_CL) +' '+ str(_LOCAL_EPOCHS)+ ' ' + str(_BATCH_SIZE) + ' ' +\
                                                        str(_LEARNING_RATE)
                                                    proc_new(total_string_to_send)
                                            else:
                                                _AGGREGATION_FACTOR=1
                                                if _ML=='CL':
                                                    for _WAITING_FACTOR_CL in waiting_factor_list:
                                                        total_string_to_send = ' ' + str(_DATE_A) + ' ' + str(
                                                            _DATE_B) + ' ' + \
                                                                               str(_DATE_C) + ' ' + str(
                                                            _T_BEGIN) + ' ' + str(_T_END) + ' ' + \
                                                                               str(_SUBDATASETS) + ' ' + str(
                                                            _DEVICES_PER_ROUND) + ' ' + str(_IID) + ' ' + str(
                                                            _EED) + ' ' + \
                                                                               str(_WIRELESS) + ' ' + str(
                                                            _MODEL) + ' ' + str(_EXPORT_CSV) + ' ' + \
                                                                               str(_USE_LOGGER) + ' ' + str(
                                                            _ML) + ' ' + str(_AGGREGATION_FACTOR) + ' ' + \
                                                                               str(_WAITING_FACTOR_CL) + ' ' + str(
                                                            _LOCAL_EPOCHS) + ' ' + str(_BATCH_SIZE) + ' ' + \
                                                                               str(_LEARNING_RATE)
                                                        proc_new(total_string_to_send)
                                                else:
                                                    _WAITING_FACTOR_CL=0
                                                    total_string_to_send = ' ' + str(_DATE_A) + ' ' + str(_DATE_B) + ' ' + \
                                                            str(_DATE_C) + ' ' + str(_T_BEGIN) + ' ' + str(_T_END) + ' ' + \
                                                            str(_SUBDATASETS) + ' ' + str(_DEVICES_PER_ROUND) + ' ' + str(_IID) + ' '+str(_EED)+' ' + \
                                                            str(_WIRELESS) + ' ' + str(_MODEL) + ' ' + str(_EXPORT_CSV) + ' ' + \
                                                            str(_USE_LOGGER) + ' ' + str(_ML) + ' ' + str(_AGGREGATION_FACTOR)  + ' ' + \
                                                            str(_WAITING_FACTOR_CL) + ' '+ str(_LOCAL_EPOCHS)+ ' ' + str(_BATCH_SIZE) + ' ' + \
                                                            str(_LEARNING_RATE)
                                                    proc_new(total_string_to_send)

run_text_total='python mymain.py'
start_time = time.time()
run()


