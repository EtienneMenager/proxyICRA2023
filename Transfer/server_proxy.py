# -*- coding: utf-8 -*-
"""Functions to create and manage a server that uses sofagym in a controller.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@inria.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2022, Inria"
__date__ = "Jul 11 2022"

from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
from socketserver import ThreadingMixIn

import socketserver
import threading
import subprocess
import queue
from os.path import dirname, abspath
import copy
import time
import os

path = dirname(dirname(abspath(__file__))) + '/'

class SimpleThreadedXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass

class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)
    def log_message(self, format, *args):
        pass

class CustomQueue(queue.Queue):
    def __init__(self):
        queue.Queue.__init__(self)
        self.entries = []

    def __str__(self):
        return str(self.entries)

    def put(self, item):
        self.entries.append(item)
        queue.Queue.put(self, item)

    def get(self, timeout=None):
        res = queue.Queue.get(self, timeout=timeout)
        self.entries.pop(0)
        return res

    def front(self):
        return self.entries[0]

    def back(self):
        return self.entries[-1]

    def __len__(self):
        return len(self.entries)

"""
instances["sofagym"]: refers to sofagym environment.
instances["inverse"]: refers to inverse scene.
"""
instances =  {"sofagym": {"pendingTasks": CustomQueue(),
                "pendingResults": CustomQueue()},
             "inverse": {"pendingTasks": CustomQueue(),
               "pendingResults": CustomQueue()},
             }

firstObservation = {"sofagym": CustomQueue(),
                   "inverse": CustomQueue()}
port_rpc = None

################################### API RPC ###################################


def getNextTask(name):
    """Distribute a pending task to the client.
    """
    res = instances[name]["pendingTasks"].get()
    return res


def taskDone(name, result):
    """Notify the server that a submitted task has been terminated by the client.
    """
    instances[name]["pendingResults"].put(result)
    return "ThankYou"


def registerFirstObservation(name, obs):
    """Function to save the first observation.
    """
    global firstObservation
    firstObservation[name].put(obs)
    return "ThankYou"


##################### API python (on the environment side) #####################

def make_action(command, **kwargs):
    """Add a command in the parameters.
    """
    m = {"command": command}
    m.update(kwargs)
    return m

def avalaible_port(to_str = False):
    """Find a free port to connect a server.
    """
    with socketserver.TCPServer(("localhost", 0), None) as s:
        free_port = s.server_address[1]

    if to_str:
        return str(free_port)
    else:
        return free_port

def start_server():
    """Start new server & first client in two dedicated threads.

       This function is not blocking and does not returns any values.
       Once the server is started it is possible to submit new tasks unsing
       the add_new_step function.It is then possible to get the results of the
       tasks using the get_results functions.
    """
    global port_rpc
    if port_rpc is None:
         port_rpc= avalaible_port()

    #Register functions
    def dispatch(port_rpc):
        with SimpleThreadedXMLRPCServer(('localhost', port_rpc), requestHandler=RequestHandler) as server:
            server.register_function(getNextTask)
            server.register_function(taskDone)
            server.register_function(registerFirstObservation)
            server.serve_forever()

    #Starts the server thread with the context.
    server_thread = threading.Thread(target=dispatch, args=(port_rpc,))
    server_thread.daemon = True
    server_thread.start()

def close_scene(names):
    """Ask the clients to close.
    """
    global instances
    for name in names:
        instances[name]["pendingTasks"].put(make_action("exit"))

    #Wait to close all clients
    time.sleep(0.01)



def start_client(name, env_infos):
    """Start the client.
    """
    global instances, port_rpc
    close_scene([name])

    #Information of the first client

    instances[name] = {"pendingTasks": CustomQueue(),
                "pendingResults": CustomQueue()}

    #Run the first client
    if name == "sofagym":
        def deferredStart():
            subprocess.run(["python3.9", path+"common/client_sofagym.py", str(port_rpc), env_infos], check=True)
    else:
        def deferredStart():
            sdict = str(env_infos)
            subprocess.run(["python3.9", path+"common/client_inverse.py",  sdict, str(port_rpc)], check=True)

    first_worker_thread = threading.Thread(target=deferredStart)
    first_worker_thread.daemon = True
    first_worker_thread.start()

    return firstObservation[name].get()


def add_action(name, todo, **kwargs):
    """Ask a client to calculate the result for a given sequence of actions.
    """
    global instances
    if name == "sofagym":
        instances[name]["pendingTasks"].put(make_action(todo, **kwargs))
    else:
        instances[name]["pendingTasks"].put(make_action(todo, **kwargs))
    return "ThankYou"


def get_result(name, timeout=None):
    """Returns available results. Blocks until a result is available.
    """
    global instances
    try:
        res = instances[name]["pendingResults"].get(timeout=timeout)
    except queue.Empty:
        print("TIMEOUT ", timeout)
        res = {}
    return res

if __name__ == "__main__":
    start_server()
    obs = start_client("sofagym", "optiabstractmultigait-v0")
    print("[INFO]  >> First observation:", obs)
    add_action("sofagym", "step", [-1, -1, -1, -1, -1])
    results = get_result("sofagym", timeout=200)
    print("[INFO]  >> Results:", results.keys())

    # add_action([-1, -1, -1, -1, 0])
    # results = get_result(timeout=200)
    # print("[INFO]  >> Results:", results.keys())
    # close_scene()
