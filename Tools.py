import os

def get_file_size_in_mb(filepath):
    size_in_bytes = os.path.getsize(filepath)
    size_in_mb=size_in_bytes/(1000*1000)
    return size_in_mb

def mbps_to_mb_per_sec(datarate):
    return datarate*0.125

