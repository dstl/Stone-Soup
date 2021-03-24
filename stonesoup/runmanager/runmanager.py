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
from stonesoup.metricgenerator.ospametric import OSPAMetric
from stonesoup.measures import Euclidean
from stonesoup.metricgenerator.manager import SimpleManager
import plotly.express as px
from _datetime import timedelta
from matplotlib import pyplot as plt

app = Flask(__name__)

#UPLOAD_FOLDER = '/path/to/the/uploads'
path= "C:/Users/gbellant.LIVAD/Documents/Projects/serapis/Stone-Soup/config.yaml"

 

@app.route('/')
def index():
    test=5
    return render_template('index.html')


@app.route('/configInput', methods=["POST","GET"])
def upload_config_input():
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
     
        #Create the tracker tree
        trackerTree = Tree(NodeData(tracker.__class__.__name__,tracker, tracker.__class__.__name__))
        trackerTree.children.append(generate_tree(tracker,trackerTree,tracker.__class__.__name__))
        

        #Create the ground truth tree
        for g in ground_truth:
          ground_truth_tree = Tree(NodeData(g.__class__.__name__,g, g.__class__.__name__))
          ground_truth_tree.children.append(generate_tree(g,ground_truth_tree,g.__class__.__name__))
                

        #Create the metric manager tree
        metric_manager_tree = Tree(NodeData(metric_manager.__class__.__name__,metric_manager, metric_manager.__class__.__name__))
        metric_manager_tree.children.append(generate_tree(metric_manager,metric_manager_tree,metric_manager.__class__.__name__))
          

        #return render_template('config.html',res = res, stateVector = stateVector, covar=covar, tracker=tracker)
        return render_template('generate.html',res = res, trackerTree=trackerTree, ground_truth_tree = ground_truth_tree,metric_manager_tree=metric_manager_tree)
        
       
      #send file name as parameter to download
    return render_template('index.html')


@app.route('/run', methods=["POST","GET"])
def run():
      if request.method == 'POST':
        config = request.form.get('configFileName')
        num_runs = int(request.form.get('num_runs'))
        with open(path, 'r') as file:
            tracker, ground_truth, metric_manager = read_config_file(file)



        trackers=[]
        ground_truths = []
        metric_managers = []

        for i in range(0,num_runs):
          tracker_copy, ground_truth_copy, metric_manager_copy = copy.deepcopy((tracker, ground_truth, metric_manager))
          trackers.append(tracker_copy)
          ground_truths.append(ground_truth_copy)
          metric_managers.append(metric_manager_copy)

        #Initialise the tracker
        tracker_copy, ground_truth_copy, metric_manager_copy = copy.deepcopy((tracker, ground_truth, metric_manager))
        tracker_min, ground_truth_min, metric_manager_min = copy.deepcopy((tracker, ground_truth, metric_manager))
        tracker_max, ground_truth_max, metric_manager_max = copy.deepcopy((tracker, ground_truth, metric_manager))
        tracker_step, ground_truth_step, metric_manager_step = copy.deepcopy((tracker, ground_truth, metric_manager))



        #Set 
        get_data(tracker_copy,tracker_min,tracker_max,tracker_step,tracker_copy.__class__.__name__,request,trackers)
        

        if(type(ground_truth)==set):
          for idx,g in enumerate(ground_truth_copy):
              get_data(g,list(ground_truth_min)[idx],list(ground_truth_max)[idx],list(ground_truth_step)[idx],g.__class__.__name__,request,ground_truths,True,idx)
        else:
              get_data(ground_truth_copy,list(ground_truth_min)[idx],list(ground_truth_max)[idx],list(ground_truth_step)[idx],ground_truth_copyg.__class__.__name__,request,ground_truths)


        metricsList=[]
        for i in range(0, num_runs):
          try:          
              print(trackers[i])
              tracks = set()
            #  print("test1")
              # print("TRACKER START")

              # print(trackers[i])
              # print("TRACKER END")

              for n, (time, ctracks) in enumerate(trackers[i],1):#, 1):
                tracks.update(ctracks)
              
              #print(tracks)
              metric_managers[i].add_data(ground_truths[i], tracks)
              
              metrics= metric_managers[i].generate_metrics()
              #print(metrics)
          except Exception as e:
              print(f'Failure: {e}', flush=True)
              #return None
          else:
              print('Success!', flush=True)
              metricsList.append(metrics)

         

        values, labels = plot(metricsList, len(metricsList))
          
        return render_template("result.html", labels=labels, values=values)


#Set the data for the plot 
def plot(metricsList, num_sims):
    metric_values = []
    time_values = []

    for metric in metricsList[0]:
      for var in metric.value:
        time_values.append(var.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'))

    for simulation in range(len(metricsList)):
      metric = []
      for m in metricsList[simulation]:
        for val in m.value:
          metric.append(val.value)
      
      metric_values.append(metric)
      #print(metric_values[simulation])
    

    return metric_values, time_values
    # for i in range(num_steps):
  
#Print the tree on the console
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
      

#Generate a tree
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
      print(object.__class__)
      parentNode.children.remove(node)
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


#Navigate inside all the tracker list given a property
def navigate_tracker(property,trackers,isSet=False,idx=0):
  newTrackers = []
  for i in range(0,len(trackers)):
    if not isSet:
      newTrackers.append(getattr(trackers[i],property))
    else:
      newTrackers.append(getattr(list(trackers[i])[idx],property))

  return newTrackers

#Navigate inside all the tracker list given a list/tuple
def navigate_tracker_list(index,trackers,isSet=False,idx=0):
  newTrackers = []
  for i in range(0,len(trackers)):
    if not isSet:
      newTrackers.append(trackers[i][index])
    else:
      newTrackers.append(list(trackers[i])[idx][index])
  return newTrackers

#Compare two tree to check if they are similar
def compare_trees(tree1,tree2,result):
 if(result==False):
  return False

 if hasattr(tree1.__class__, "properties"):
    properties = tree1.__class__.properties
    if len(properties)>0:
      for property in properties:
      #  trackers = get_data(getattr(object,property),getattr(object_min,property),getattr(object_max,property),getattr(object_steps,property),propertyName+"."+property,request,gen_object_array(getattr(object,property),len(trackers)))
        result = compare_trees(getattr(tree1,property),getattr(tree2,property),result)
    else:
        if tree1==tree2:
          print("TRUE")
          return True
        else:
          print(type(tree1))
          print(type(tree2))
          print("tree1 "+str(tree1)+" tree2 "+str(tree2))
          return True
 else:
      if(type(tree1) is list or type(tree1) is tuple):
        for i,el in enumerate(tree1):
          
          #trackers =get_data(el,object_min[i],object_max[i],object_steps[i],propertyName+'['+str(i)+']',request,gen_object_array(el,len(trackers)))
          try:  
            result = compare_trees(el,tree2[i],result)
          except:
            print("el "+str(el)+" tree2 "+str(tree2[i]))
            return False
      else: 
        className = tree1.__class__.__name__        
        if (className=="ndarray" or className=="StateVector" or className == "CovarianceMatrix"):
            if(tree1==tree2).all():
              print("TRUE")
              return True
            else:
              print("abc")
              print(className)
              print("tree1 "+str(tree1)+" tree2 "+str(tree2))
              return False
        elif (className=="timedelta" or className=="datetime" or className=="int" or className == "float" or className=="bool" or className=="NoneType"):
          if(tree1==tree2):
              return True
          else:
              print("tree1 "+str(tree1)+" tree2 "+str(tree2))
              return False
        else:
          print(className)
          return False
 return result
#return trackers

#Get the data from the POST request
def get_data(object,object_min,object_max,object_steps, propertyName,request,trackers,isSet=False,idx=0):
  if hasattr(object.__class__, "properties"):
    properties = object.__class__.properties
    if len(properties)>0:
      for single_property in properties:
        new_trackers = get_data(getattr(object,single_property),getattr(object_min,single_property),getattr(object_max,single_property),getattr(object_steps,single_property),propertyName+"."+single_property,request,navigate_tracker(single_property,trackers,isSet,idx))
    else: 
      return trackers
  else:
      if(type(object) is list or type(object) is tuple):
        for i,el in enumerate(object):
          new_trackers = get_data(el,object_min[i],object_max[i],object_steps[i],propertyName+'['+str(i)+']',request,navigate_tracker_list(i,trackers))

      else: 
        set_tracker_data(object,object_min, object_max, object_steps,propertyName,request,trackers)
        
#Set the data value inside the trackers
def set_tracker_data(object,object_min,object_max,object_steps,type,request,trackers):

  if(object.__class__.__name__=='StateVector'):
     set_state_vector(object,object_min,object_max,object_steps,type,request,trackers)
  elif(object.__class__.__name__=='CovarianceMatrix'):
    set_covar(object,object_min,object_max,object_steps,type,request,trackers)
  elif(object.__class__.__name__=='datetime' ):
    set_datetimes(object,object_min,object_max,object_steps,type,request,trackers)
  elif(object.__class__.__name__=='ndarray'):
    set_nd_array(object,object_min,object_max,object_steps,type,request,trackers)
  elif(object.__class__.__name__=='int' or object.__class__.__name__=='float' or object.__class__.__name__=='NoneType' or object.__class__.__name__=='bool' ):
    set_single_value(object,object_min,object_max,object_steps,type,request,trackers)


def set_bool(object,objectMin,objectMax,objectSteps, type,request,trackers):
  objectType = object.__class__.__name__

  object = request.form.get(type)
  objectMin = request.form.get(type+'_min_range')
  objectMax = request.form.get(type+'_max_range')
  objectSteps = request.form.get(type+'_step')
  
  for k in range(0,len(trackers)):
        trackers[k] = bool(object)
     

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
      elif(objectType=='bool'):
          trackers[k] = bool(bool(object))
      else:
        if(object=="None"):
          trackers[k]=None
        else:
          trackers[k] = object


def set_datetimes(object,objectMin,objectMax,objectSteps, type,request,trackers):

  object = request.form.get(type)
  objectMin = request.form.get(type+'_min_range')
  objectMax = request.form.get(type+'_max_range')
  objectSteps = request.form.get(type+'_step')
  for k in range(0,len(trackers)):
        if object == None:
          trackers[k]= object
        else:
          trackers[k] = datetime.datetime.strptime(object, '%Y-%m-%d %H:%M:%S.%f')


def generate_random(val,valMin,valMax,valSteps,index_run):
  if(valSteps!=0 ):
    val = valMin + valSteps*index_run
    if val>valMax:
      return valMax
    else:
       return float(val)
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
       return int(val)
  elif valMin<valMax:
    return random.randint(valMin,valMax)
  else: 
    return int(val)


if __name__== "__main__":
    app.run(debug=True)

