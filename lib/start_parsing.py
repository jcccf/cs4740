
from subprocess import Popen
from math import ceil
import os,sys,time

servers = []
clients = []
nprocs = 8
base_port = 8080
lb,ub = 201,400
bounds = range(lb,ub,int(ceil( float(ub-lb)/nprocs ))) + [ub]
print bounds
assert len(bounds) == nprocs+1
# exit(0)

try:
    #start up servers
    saved_dir = os.getcwd()
    os.chdir(os.path.join(saved_dir,"..","corenlp"))
    print os.getcwd()
    
    for i in range(nprocs):
        servers.append( Popen("python corenlp.py -p %d"%(base_port+i), shell=True) )
        
    os.chdir(saved_dir)
    print os.getcwd()
    
    #wait for some time
    time.sleep(30.0) #30 secs
    
    #start up clients
    for i in range(nprocs):
        clients.append( Popen(
            "python CoreNLPLoader.py --port %d -l %d -u %d"%(base_port+i, bounds[i],bounds[i+1]),
            shell=True) )
    
    # wait for every client to be done
    for i in range(nprocs):
        clients[i].wait()
    
    # kill everything
    for i in range(nprocs):
        try:
            servers[i].kill()
        except:
            pass
    for i in range(nprocs):
        try:
            client[i].kill()
        except:
            pass
        
except Exception as e:
    print e
    # kill everything
    for i in range(nprocs):
        try:
            servers[i].kill()
        except:
            pass
    for i in range(nprocs):
        try:
            client[i].kill()
        except:
            pass

