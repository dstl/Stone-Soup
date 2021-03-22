import os
import sys
from flask import Flask, render_template,url_for,request,redirect
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap
from stonesoup.runmanager.parameters import Parameters
from stonesoup.serialise import YAML
from stonesoup.types.array import CovarianceMatrix
from numpy.random.mtrand import randint
from stonesoup.runmanager import RunManager
import numpy as np
from stonesoup.runmanager.utils import generate_covar, generate_random_array, generate_random_covar, generate_random_int, Tree,NodeData
from stonesoup.initiator.simple import GaussianParticleInitiator,SinglePointInitiator
from stonesoup.platform.base import MovingPlatform, MultiTransitionMovingPlatform, Platform
from stonesoup.base import Base
from anytree import Node
import random
import inspect
import importlib
import pkgutil
import warnings
from stonesoup.tracker.base import Tracker
import copy
import datetime
app = Flask(__name__)

#UPLOAD_FOLDER = '/path/to/the/uploads'
path= "C:/Users/gbellant.LIVAD/Documents/Projects/serapis/Stone-Soup/config.yaml"

 

@app.route('/')
def index():
    test=5
    return render_template('index.html')


@app.route('/configInput', methods=["POST","GET"])
def upload_config_input():
    variable="5"
    if request.method == 'POST':

        #configFile = request.form["configFile"]
        # check if the post request has the file part
        if 'configFile' not in request.files:
            print('no file')
            print(request.url)
            return render_template('index.html')
        
        
        configFile = request.files['configFile']

        # if user does not select file, browser also
        # submit a empty part without filename

       # res.num_runs = request.form('num_runs')
        #res.num_processes = request.form('num_processes')
        #res.output = request.form('filename')

        if configFile.filename == '':
            print('no filename')
            return render_template('index.html')
        else:
            filename = secure_filename(configFile.filename)
            #configFile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("saved file successfully")
        
        res = Parameters()
        res.filename = configFile.name
        res.num_runs = request.form['num_runs']
        res.num_processes = request.form['num_processes']
        res.output = request.form['filename']     
        tracker, ground_truth, metric_manager = read_config_file(configFile)
        runManager = RunManager()
        runManager.tracker = tracker
        #print(Base.subclasses)
#       
#        #import_submodules()
      #  generate_initiator(tracker)
        
        #parentList = generate_template_tree(tracker.initiator,[])
        parentList = Tree(NodeData(tracker.__class__.__name__,tracker, tracker.__class__.__name__))
        parentList.children.append(generate_tree(tracker,parentList,tracker.__class__.__name__))
        #print("TEST")
        #print(printTree(parentList.data, parentList.children))

        #return render_template('config.html',res = res, stateVector = stateVector, covar=covar, tracker=tracker)
        return render_template('generate.html',res = res, parentList=parentList)
        
       
      #send file name as parameter to download
    return render_template('index.html')



@app.route('/run', methods=["POST","GET"])
def run():
      if request.method == 'POST':
        config = request.form.get('configFileName')
        num_runs = int(request.form.get('num_runs'))
        with open(path, 'r') as file:
            tracker, ground_truth, metric_manager = read_config_file(file)


        #print(type(tracker.detector.platforms[0].transition_model.model_list[0]))

        trackers=[]
        for i in range(0,num_runs):
          trackers.append(copy.deepcopy(tracker))

        trackers = get_data(tracker,copy.deepcopy(tracker),copy.deepcopy(tracker),copy.deepcopy(tracker),tracker.__class__.__name__,request,trackers)

        # for idx, tr in enumerate(trackers):
        #    print("tracker "+str(idx))
        #    print(tracker)
        metricsList=[]

        print(trackers[0])
        if(tracker==trackers[0]):
          print("YES")
        for i in range(0, 1):
          try:
              tracks = set()
            #  print("test1")

              #print(trackers[i])
              for n, (time, ctracks) in enumerate(trackers[i], 1):
                  tracks.update(ctracks)

              print("test2")

              metric_manager.add_data(ground_truth, tracks)

              metrics= metric_manager.generate_metrics()
              #print(metrics)
              metricsList.append(metrics)
          except Exception as e:
              print(f'Failure: {e}', flush=True)
              return None
          else:
              print('Success!', flush=True)
         

                  

    
         
        return "metricsList"


def printTree(data, treeNodes,deep=-1):
  deep=deep+1
  print(str(data.property )+' Deep '+str(deep))
  deep=deep+1
  for node in treeNodes:
    if(hasattr(node,"children")):
      if(len(node.children)>0):
        printTree(node.data,node.children,deep-1)
      else: 
        print(str(node.data.property)+ ' Deep '+str(deep))
      

def generate_tree(object,parentNode,propertyName):
  if(type(object) is not list and type(object) is not tuple):
    node:Tree = Tree(NodeData( object.__class__.__name__, object,propertyName))

    parentNode.children.append(node)

  if hasattr(object.__class__, "properties"):
    properties = object.__class__.properties
    if len(properties)>0:
      for property in properties:    
         generate_tree(getattr(object,property),node,property)

    else: 
      return node
  else:
      if(type(object) is list or type(object) is tuple):
        for i,el in enumerate(object):
          generate_tree(el,parentNode,propertyName+'['+str(i)+']')

      else: 
        return node


def read_config_file(config_file): 
    config_string = config_file.read()
    tracker, ground_truth, metric_manager = YAML('safe').load(config_string)
    return tracker, ground_truth, metric_manager


def gen_object_array(object,num_runs):
  trackers=[]
  for i in range(0,num_runs):
            trackers.append(copy.deepcopy(object))

  return trackers


def navigate_tracker(value,trackers):
  newTrackers = []
  for i in range(0,len(trackers)):
    newTrackers.append(value)
  return newTrackers


def updateTracker(property,old_trackers,new_trackers):
  for i in range(0,len(old_trackers)):
    try:
      setattr(old_trackers[i], property,new_trackers[i])
     # print("Set ATT")
    except: 
      return old_trackers
  return old_trackers


def updateTrackerList(old_trackers,new_trackers,index):
  for i in range(0,len(old_trackers)):
      if(type(new_trackers[i] is  tuple)):
        trackList = copy.deepcopy(list(old_trackers[i]))
      else:
        trackList = copy.deepcopy(old_trackers[i])
         
      #for j in range(0,len(old_trackers[i])):
      trackList[index] = copy.deepcopy(new_trackers[i])
      # if(type(new_trackers[i] is  tuple)):
      #   print(new_trackers[i].__class__.__name__)
      #   old_trackers[i] = tuple(trackList)
      # else:
      old_trackers[i] = trackList
        


     # print("Set ATT")
  return old_trackers


def get_data(object,object_min,object_max,object_steps, propertyName,request,trackers):
 # if(type(object) is not list and type(object) is not tuple):
    #node:Tree = Tree(NodeData( object.__class__.__name__, object,propertyName))
    #print(node.data.property)
    #parentNode.children.append(node)

  #object:Tracker
  if hasattr(object.__class__, "properties"):
    properties = object.__class__.properties
    if len(properties)>0:
      for property in properties:
      #  trackers = get_data(getattr(object,property),getattr(object_min,property),getattr(object_max,property),getattr(object_steps,property),propertyName+"."+property,request,gen_object_array(getattr(object,property),len(trackers)))
        old_trackers = copy.deepcopy(trackers)    
        trackers = get_data(getattr(object,property),getattr(object_min,property),getattr(object_max,property),getattr(object_steps,property),propertyName+"."+property,request,navigate_tracker(getattr(object,property),trackers))
        trackers = updateTracker(property, old_trackers,copy.deepcopy(trackers))

    else: 
      return trackers
  else:
      if(type(object) is list or type(object) is tuple):
        for i,el in enumerate(object):
          #trackers =get_data(el,object_min[i],object_max[i],object_steps[i],propertyName+'['+str(i)+']',request,gen_object_array(el,len(trackers)))
          old_trackers = copy.deepcopy(trackers)    
          trackers = get_data(el,object_min[i],object_max[i],object_steps[i],propertyName+'['+str(i)+']',request,navigate_tracker(el,trackers))
          trackers = updateTrackerList(old_trackers,trackers,i) 

      else: 
        object,object_min,object_max,object_steps,trackers = set_tracker_data(object,object_min, object_max, object_steps,propertyName,request,trackers)
        
        return trackers

  return trackers


def set_tracker_data(object,object_min,object_max,object_steps,type,request,trackers):

  if(object.__class__.__name__=='StateVector'):
    object,object_min,object_max,object_steps,trakers = set_state_vector(object,object_min,object_max,object_steps,type,request,trackers)
  elif(object.__class__.__name__=='CovarianceMatrix'):
    object,object_min,object_max,object_steps,trackers = set_covar(object,object_min,object_max,object_steps,type,request,trackers)
  elif(object.__class__.__name__=='datetime'):
    object,object_min,object_max,object_steps,trackers = set_datetimes(object,object_min,object_max,object_steps,type,request,trackers)
  elif(object.__class__.__name__=='ndarray'):
    object,object_min,object_max,object_steps,trackers = set_nd_array(object,object_min,object_max,object_steps,type,request,trackers)
  elif(object.__class__.__name__=='int' or object.__class__.__name__=='float' or object.__class__.__name__=='NoneType' or object.__class__.__name__=='bool' ):
      object,object_min,object_max,object_steps,trackers = set_single_value(object,object_min,object_max,object_steps,type,request,trackers)

  return object,object_min,object_max,object_steps,trackers

#STATE VECTOR
def set_state_vector(object,objectMin,objectMax,objectSteps, type,request,trackers):
  
  state_vector = request.form.getlist(type+'[]')
  state_vector_min = request.form.getlist(type+'_min_range[]')
  state_vector_max = request.form.getlist(type+'_max_range[]')
  state_vector_step = request.form.getlist(type+'_step[]')
  


  for i,val in enumerate(state_vector):
     object[i] =  val
     objectMin[i] = state_vector_min[i]
     objectMax[i] = state_vector_max[i]
     objectSteps[i] = state_vector_step[i]


     for k in range(0,len(trackers)):
       trackers[k][i]= generate_random(object[i],objectMin[i],objectMax[i],objectSteps[i],k)

  return object,objectMin,objectMax,objectSteps,trackers


def set_covar(object,objectMin,objectMax,objectSteps, type,request,trackers):
  
  covar = request.form.getlist(type+'[]')
  covar_min = request.form.getlist(type+'_min_range[]')
  covar_max = request.form.getlist(type+'_max_range[]')
  covar_step = request.form.getlist(type+'_step[]')
  #print(covar)
  for i in range(0,len(object)):
    for j in range(0,len(object)):
      object[i,j] = covar[j+len(object)*i]
      objectMin[i,j] = covar_min[j+len(object)*i]
      objectMax[i,j] = covar_max[j+len(object)*i]
      objectSteps[i,j] = covar_step[j+len(object)*i]
      
      for k in range(0,len(trackers)):
       trackers[k][i,j]= generate_random(object[i,j],objectMin[i,j],objectMax[i,j],objectSteps[i,j],k)
      
  return object,objectMin,objectMax,objectSteps,trackers


def set_nd_array(object,objectMin,objectMax,objectSteps, type,request,trackers):
  
  nd_array = request.form.getlist(type+'[]')
  nd_array_min = request.form.getlist(type+'_min_range[]')
  nd_array_max = request.form.getlist(type+'_max_range[]')
  nd_array_step = request.form.getlist(type+'_step[]')
  
  for i,val in enumerate(nd_array):
     object[i] = val
     objectMin[i] = nd_array_min[i]
     objectMax[i] = nd_array_max[i]
     objectSteps[i] = nd_array_step[i]


  return object,objectMin,objectMax,objectSteps,trackers 


def set_single_value(object,objectMin,objectMax,objectSteps, type,request,trackers):
  objectType = object.__class__.__name__

  object = request.form.get(type)
  objectMin = request.form.get(type+'_min_range')
  objectMax = request.form.get(type+'_max_range')
  objectSteps = request.form.get(type+'_step')
  
  for k in range(0,len(trackers)):
      if(objectType=="int"):
        trackers[k] = int(generate_random_int(int(object),int(objectMin),int(objectMax),int(objectSteps),k))
      elif(objectType=='float'):
        try:
          trackers[k] =float(generate_random(float(object),float(objectMin),float(objectMax),float(objectSteps),k))
        except ValueError:
          trackers[k]=0.0
      else:
        if(object=="None"):
          trackers[k]=None
        else:
          trackers[k] = object

  return object,objectMin,objectMax,objectSteps,trackers


def set_datetimes(object,objectMin,objectMax,objectSteps, type,request,trackers):

  object = request.form.get(type)
  objectMin = request.form.get(type+'_min_range')
  objectMax = request.form.get(type+'_max_range')
  objectSteps = request.form.get(type+'_step')

  for k in range(0,len(trackers)):
        trackers[k] = datetime.datetime.strptime(object, '%Y-%m-%d %H:%M:%S.%f')


  return object,objectMin,objectMax,objectSteps,trackers

def generate_random(val,valMin,valMax,valSteps,index_run):
  if(valSteps!=0 ):
    val = valMin + valSteps*index_run
    if val>valMax:
      return valMax
    else:
       return val
  elif valMin<valMax:
    return random.uniform(valMin,valMax)
  else: 
    return float(val)


def generate_random_int(val,valMin,valMax,valSteps,index_run):
  if(valSteps!=0 ):
    val = valMin + valSteps*index_run
    if val>valMax:
      return valMax
    else:
       return val
  elif valMin<valMax:
    return random.randint(valMin,valMax)
  else: 
    return int(val)


if __name__== "__main__":
    app.run(debug=True)

