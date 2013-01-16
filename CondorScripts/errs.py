#!/usr/bin/python
# Functions used to fetch all the erros 
# Alborz Geramifard 2013 MIT
# Assumes linux machine just for clear screen! Why do you want to run it on something else anyway?

#Inputs:
# idir : Initial Directory
import os, sys, time, re 
from Script_Tools import *

def fetchErrs(idir,detailed):
        if not os.path.exists(idir+'/main.py'):
            #Not a task directory
            for folder in os.listdir(idir):
                if os.path.isdir(idir+'/'+folder) and folder[0] != '.':
                    fetchErrs(idir+'/'+folder,detailed)
        else:                
            errids = set()
            jobs = glob.glob(idir+'/CondorOutput/err/*.err')
            for job in jobs:
                _,_,jobname = job.rpartition('/')
                errid,_,_ = jobname.rpartition('.')
                errids.add(eval(errid))

            total       = len(glob.glob(idir+'/CondorOutput/log/*.log'))
            errs        = len(errids);

            if errs:
                print"%s: %s%d%s/%s%d%s"  % (idir.replace('./',''),RED,errs,NOCOLOR,TOTAL_COLOR,total,NOCOLOR)
            if detailed:
                logs = ''
                for errid in errids:
                    command = "tail -n 30 " + idir+'/CondorOutput/err/%d.err' % errid
                    sysCommandHandle = os.popen(command)
                    empty = True
                    for line in sysCommandHandle:
                        print RED+"#%02d: %s"  % (errid, line.rstrip('\n\r')) + NOCOLOR
                        empty = False
                    if empty:
                        print RED+"#%02d: Empty Err File!\n"  % (errid)+ NOCOLOR
if __name__ == '__main__':
    os.system('clear');
    print('*********************************************************');    
    print('************** Reporting For Duty! **********************');    
    print('*********************************************************');    
    detailed = len(sys.argv) > 1
    fetchErrs('.',detailed)   
    
