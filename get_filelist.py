import numpy as np
from pathlib import Path, PurePath
from time import time, sleep
import pp
import multiprocessing
import os
import argparse
import gc

gc.enable()

np.set_printoptions(threshold=np.inf)
filelist = []
for filename in Path('music').glob('**/*.mp3'):
    filelist.append(filename)
for filename in Path('music').glob('**/*.wav'):
    filelist.append(filename)  

print len(filelist)
#print(filelist)

def parallel_python_process(process_id, cpu_filelist):
    import numpy as np
    from pathlib import Path, PurePath
    from time import time, sleep
    import pp
    import multiprocessing
    import os
    import argparse
    import gc
    count = 1
    for file_name in cpu_filelist:
        path = str(PurePath(file_name))
        filename = path.replace(".","").replace(";","").replace(",","").replace("mp3",".mp3").replace("aiff",".aiff").replace("aif",".aif").replace("au",".au").replace("m4a", ".m4a").replace("wav",".wav").replace("flac",".flac").replace("ogg",".ogg")  # rel. filename as from find_files
        print ("File " + path + " " + str(count) + " von " + str(len(cpu_filelist)))               
        with open("features0/out.files", "a") as myfile:
            print ("File " + path + " " + str(count) + " von " + str(len(cpu_filelist))) 
            line = (filename + "     :       " + str(process_id))
            myfile.write(line + '\n')       
            myfile.close()
     
        count = count + 1
        gc.enable()
        gc.collect()
    
    # Perform any action like print a string
    #print("calculating this takes ...")

    # Store end time
    #end_time = time.time()

    # Calculate the execution time and print the result
    #print("%.10f seconds" % (end_time - start_time))
    
    gc.enable()
    gc.collect()
    return 1
    #return (end_time - start_time)


def process_stuff (startjob, maxparts):
    import numpy as np
    from pathlib import Path, PurePath
    from time import time, sleep
    import pp
    import multiprocessing
    import os
    import argparse
    import gc
    print("Init done")   
    print """Usage: python extract_in_parallel.py"""
    #sys.path.remove('/usr/share/pyshared')
    start = 0
    end = len(filelist)
    cpus = multiprocessing.cpu_count()
    print("Detected cores: ")
    print cpus
    ncpus = (cpus / 2) - 1
    ncpus = 4
    print("Used cores: ")
    print ncpus
    print("files per part: ")
    files_per_part = 25
    #files_per_part = 200
    print(files_per_part)
    # Divide the task into subtasks - such that each subtask processes around 4 songs
    parts = (len(filelist) / files_per_part) + 1
    print("Split problem in parts: ")
    print parts

    with open("features0/out.files", "w") as myfile:
        myfile.write("")
        myfile.close()

    startjob = int(startjob)
    maxparts = int(maxparts)

    print("starting with")    
    print(startjob)
    print("ending with")
    print(maxparts)

    #if parts < maxparts:
    if startjob != 0 or maxparts != parts:
        #parts = ncpus
        step = (end - start) / parts + 1
        # Create jobserver
        job_server = pp.Server()
        # Execute the same task with different amount of active workers and measure the time
        #for ncpus in (1, 2, 4, 8, 16, 1):
        job_server.set_ncpus(ncpus)
        jobs = []
        #parallel_python_process(1, filelist, 1, 1, 1, 1, 1)
        print "Starting ", job_server.get_ncpus(), " workers"
        #for index in xrange(startjob, startjob + maxparts):
        if maxparts > parts:
            maxparts = parts
        for index in xrange(startjob, maxparts):
            #not <= (range(startjob, parts) would go to parts-1 as well)
            if index < parts:        
                starti = start+index*step
                endi = min(start+(index+1)*step, end)
                #print index
                #print starti
                #print endi    
                #PARAMS: filelist, mfcc_kl, mfcc_euclid, notes, chroma, bh    
                jobs.append(job_server.submit(parallel_python_process, (index, filelist[starti:endi])))
                gc.collect()
        # Retrieve all the results and calculate their sum
        times = sum([job() for job in jobs])
        #print(times / ncpus)
        # Print the partial sum
        #print "Partial sum is", part_sum1, "| diff =", math.log(2) - part_sum1
        job_server.print_stats()
    else:
        #parts = ncpus
        step = (end - start) / parts + 1
        # Create jobserver
        job_server = pp.Server()
        # Execute the same task with different amount of active workers and measure the time
        #for ncpus in (1, 2, 4, 8, 16, 1):
        job_server.set_ncpus(ncpus)
        jobs = []
        #parallel_python_process(1, filelist, 1, 1, 1, 1, 1)
        print "Starting ", job_server.get_ncpus(), " workers"
        #can continue previously started jobs: 
        startjob = 0
        for index in xrange(startjob, parts):
            starti = start+index*step
            endi = min(start+(index+1)*step, end)
            #print index
            #print starti
            #print endi    
            #PARAMS: filelist, mfcc_kl, mfcc_euclid, notes, chroma, bh    
            jobs.append(job_server.submit(parallel_python_process, (index, filelist[starti:endi])))
            gc.collect()
        # Retrieve all the results and calculate their sum
        times = sum([job() for job in jobs])
        #print(times / ncpus)
        # Print the partial sum
        #print "Partial sum is", part_sum1, "| diff =", math.log(2) - part_sum1
        job_server.print_stats()
    #if more than 200 RAM will flow regardless of GC
    del job_server
    del jobs
    gc.enable()
    gc.collect()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('start', help='start ID', default=0)
    argparser.add_argument('end', help='end ID', default=np.inf) 
    args = argparser.parse_args()

    do_mfcc_kl = 1
    do_mfcc_euclid = 1
    do_notes = 1
    do_chroma = 1
    do_bh = 1
    startjob = 0
    maxparts = 400

    # BATCH FEATURE EXTRACTION:
    process_stuff(args.start,args.end)

