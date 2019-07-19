#!/usr/bin/env python
from __future__ import print_function
from mpi4py import MPI
import numpy as np
from pathlib import Path, PurePath
from time import time, sleep
import multiprocessing
import os
import argparse
import gc

# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object

gc.enable()
filelist = []
for filename in Path('music').glob('**/*.mp3'):
    filelist.append(filename)
for filename in Path('music').glob('**/*.wav'):
    filelist.append(filename)  
print("length of filelist" + str(len(filelist)))

def parallel_python_process(process_id, cpu_filelist):
    print("calling rank " + str(rank) + " size " + str(size))
    count = 1
    for file_name in cpu_filelist:
        path = str(PurePath(file_name))
        filename = path.replace(".","").replace(";","").replace(",","").replace("mp3",".mp3").replace("aiff",".aiff").replace("aif",".aif").replace("au",".au").replace("m4a", ".m4a").replace("wav",".wav").replace("flac",".flac").replace("ogg",".ogg")  # rel. filename as from find_files
        #print ("File " + path + " " + str(count) + " von " + str(len(cpu_filelist)))               
        with open("features0/out" + str(rank) + ".files", "a") as myfile:
            #print ("File " + path + " " + str(count) + " von " + str(len(cpu_filelist))) 
            line = (filename + "     :       " + str(process_id))
            myfile.write(line + '\n')       
            myfile.close()
        count = count + 1
        gc.enable()
        gc.collect()
    gc.enable()
    gc.collect()
    return 1

def process_stuff():
    startjob = 0        
    start = 0
    end = len(filelist)
    print("used cores: " + str(size))
    ncpus = size
    print("files per part: ")
    files_per_part = 25
    print(files_per_part)
    # Divide the task into subtasks - such that each subtask processes around 25 songs
    parts = (len(filelist) / files_per_part) + 1
    print("Split problem in parts: ")
    print(str(parts))
    with open("features0/out" + str(rank) + ".files", "w") as myfile:
        myfile.write("")
        myfile.close()
    step = (end - start) / parts + 1
    for index in xrange(startjob + rank, parts, size):
        if index < parts:        
            starti = start+index*step
            endi = min(start+(index+1)*step, end)
            print("calling rank " + str(rank) + " size " + str(size) + " starti " + str(starti) + " endi " + str(endi))
            parallel_python_process(index, filelist[starti:endi])
            gc.collect()
    gc.enable()
    gc.collect()

process_stuff()
