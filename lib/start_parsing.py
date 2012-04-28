
from subprocess import Popen
from math import ceil
import os,sys,time
import argparse

servers = []
clients = []
nprocs = 1
base_port = 8080
lb,ub = 201,400
bounds = range(lb,ub,int(ceil( float(ub-lb)/nprocs ))) + [ub]
print bounds
assert len(bounds) == nprocs+1
# exit(0)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-s', action='store_true', dest="s", help="starts the server(s)")
    argparser.add_argument('-c', action='store_true', dest="c", help="starts the client(s)")
    args = argparser.parse_args()
    try:
        if args.s:
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
        
        if args.c:
            #start up clients
            for i in range(nprocs):
                clients.append( Popen(
                    "python CoreNLPLoader.py --port %d -l %d -u %d > %s"%(
                        base_port+i, bounds[i], bounds[i+1]-1, "out_%d.txt"%i),
                        shell=True) )
        
        # wait for every client to be done
        for c in clients:
            c.wait()
            
        # wait for every server to be done
        for s in servers:
            s.wait()
        
        # kill everything
        for s in servers:
            try:
                s.kill()
            except:
                pass
        for c in clients:
            try:
                c.kill()
            except:
                pass
            
    except Exception as e:
        print e
        # kill everything
        for s in servers:
            try:
                s.kill()
            except:
                pass
        for c in clients:
            try:
                c.kill()
            except:
                pass

