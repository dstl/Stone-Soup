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

import inspect
import importlib
import pkgutil
import warnings


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
        stateVector = tracker.initiator.initiator.prior_state.state_vector
        covar = tracker.initiator.initiator.prior_state.covar
        test = 5
      #  generate_initiator(tracker)
        var = ""
        
        #parentList = generate_template_tree(tracker.initiator,[])
        parentList = Tree(NodeData(tracker.__class__.__name__,tracker, tracker.__class__.__name__))
        parentList.children.append(generate_tree(tracker,parentList,tracker.__class__.__name__))
        #print("TEST")
        #print(printTree(parentList.data, parentList.children))

        #return render_template('config.html',res = res, stateVector = stateVector, covar=covar, tracker=tracker)
        return render_template('generate.html',res = res, parentList=parentList)
        
       
      #send file name as parameter to download
    return render_template('index.html')

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

  print(str(propertyName)+" type "+str(type(object)))

  if(type(object) is not list and type(object) is not tuple):
    node:Tree = Tree(NodeData( object.__class__.__name__, object,propertyName))
    #print(node.data.property)
    parentNode.children.append(node)

  
  #print(parentNode.data.property)
  if hasattr(object.__class__, "properties"):
    properties = object.__class__.properties
    if len(properties)>0:
      for property in properties:    
         generate_tree(getattr(object,property),node,property)
         #print(node.data.property) 
    else: 
      return node
  else:
      if(type(object) is list or type(object) is tuple):
        for el in object:
          generate_tree(el,parentNode,propertyName)
          #print(node.data.property) 
      else: 
        return node


def generate_template_tree(object,parentList):
  component=[]
  if hasattr(object.__class__, "properties"):
    properties = object.__class__.properties
    if len(properties)>0:
      for property in properties:    
        parentList.append(object.__class__.__name__)
        generate_template_tree(getattr(object,property),parentList)
        parentList.remove(object.__class__.__name__)
        #print(property)
      # print(getattr(detector,property))
    else: 
      parentList.append(object.__class__.__name__)
      print(object.__class__.__name__)      
      parentList.remove(object.__class__.__name__)
      return parentList

  else:
      if(type(object) is list):
        for el in object:
          parentList.append(object.__class__.__name__)
          generate_template_tree(el,parentList)
          parentList.remove(object.__class__.__name__)

      else: 
        parentList.append(object.__class__.__name__)
        #return parentList
        print(parentList)
        parentList.remove(object.__class__.__name__)
        return parentList

        

def read_config_file(config_file): 
    config_string = config_file.read()
    tracker, ground_truth, metric_manager = YAML('safe').load(config_string)
    return tracker, ground_truth, metric_manager


@app.route('/run', methods=["POST","GET"])
def run():
      if request.method == 'POST':
        config = request.form.get('configFileName')
        num_runs = request.form.get('num_runs')
        with open(path, 'r') as file:
            tracker, ground_truth, metric_manager = read_config_file(file)


        #print(type(tracker.detector.platforms[0].transition_model.model_list[0]))
        runManager = get_data(tracker)
        n_run = 4


        vectorLen = len(runManager.state_vectors[0])
        stateVector=[]

        for i in range(0, n_run-1):
            #STATE VECTOR
            runManager.state_vectors.append(generate_random_array(runManager.state_vector_min_range,runManager.state_vector_max_range,vectorLen))
            #COVAR 
            runManager.covar.append(generate_random_covar(runManager.covar_min_range,runManager.covar_max_range,vectorLen))
            #NUMBER PARTICLES
            runManager.number_particles.append(generate_random_int(runManager.number_particles_min_range,runManager.number_particles_max_range))
            #TIME STEPS SINCE UPDATE
            runManager.time_steps_since_update.append(generate_random_int(runManager.time_steps_since_update_min_range,runManager.time_steps_since_update_max_range))



        print(runManager.time_steps_since_update)

        return "OK"


def get_data(tracker):
      runManager = RunManager()

      runManager = initialise_tracker(runManager,tracker.initiator)
      
      runManager = initialise_deleter(runManager)

      initialise_detector(tracker,[])

      return runManager



def initialise_tracker(runManager,tracker):
      if tracker.__class__== GaussianParticleInitiator:
        #NUMBER PARTICLES
        runManager.number_particles.append(request.form.getlist('number_particles')[0])
        runManager.number_particles_min_range = request.form.getlist('number_particles_min_range')[0]
        runManager.number_particles_max_range = request.form.getlist('number_particles_max_range')[0]
        runManager.number_particles_step = request.form.getlist('number_particles_step')[0]

        if tracker.initiator.__class__ == SinglePointInitiator:
            #GET STATE VECTOR
            runManager.state_vectors.append( request.form.getlist('state_vector[]'))
            runManager.state_vector_min_range = request.form.getlist('state_vector_min_range[]')
            runManager.state_vector_max_range = request.form.getlist('state_vector_max_range[]')
            runManager.state_vector_step = request.form.getlist('state_vector_step[]')
            
            lenght = len(runManager.state_vectors[0])
            #GET COVAR
            runManager.covar.append(generate_covar(request.form.getlist('covar[]'),lenght))
            runManager.covar_min_range.append(generate_covar(request.form.getlist('covar_min_range[]'),lenght ))
            runManager.covar_max_range.append(generate_covar(request.form.getlist('covar_max_range[]'),lenght ))
            runManager.covar_step.append(generate_covar(request.form.getlist('covar_step[]'),lenght ))



      return runManager

def initialise_deleter(runManager):
       #NUMBER PARTICLES
      runManager.time_steps_since_update.append(request.form.getlist('time_steps_since_update')[0])
      runManager.time_steps_since_update_min_range = request.form.getlist('time_steps_since_update_min_range')[0]
      runManager.time_steps_since_update_max_range = request.form.getlist('time_steps_since_update_max_range')[0]
      runManager.time_steps_since_update_step = request.form.getlist('time_steps_since_update_step')
      
      #print(request.form.getlist('time_steps_since_update_max_range'))
      return runManager


def initialise_detector(detector,  parentList):
  if hasattr(detector.__class__, "properties"):
    properties = detector.__class__.properties
    if len(properties)>0:
      for property in properties:    
        parentList.append(detector.__class__.__name__)
        initialise_detector(getattr(detector,property),parentList)
        parentList.remove(detector.__class__.__name__)
        #print(property)
      # print(getattr(detector,property))
    else: 
      parentList.append(detector.__class__.__name__)
      #print("WITH PROPERTIES")
      #print(detector.__class__.__name__)
      #parentList.remove(detector.__class__.__name__)

  else:
      if(type(detector) is list):
        for el in detector:
          parentList.append(detector.__class__.__name__)
          initialise_detector(el,parentList)
          parentList.remove(detector.__class__.__name__)

      else: 
        print("WITHOUT")
        parentList.append(detector.__class__.__name__)
        print(parentList)
        parentList.remove(detector.__class__.__name__)
 


def initialise_detector_temp(runManager:RunManager,detector):
    #Platforms
      flag=True
      platform_index = 0
    
      print(detector.subclasses)
      for platform in detector.platforms:
        platform_index=platform_index+1
        #if platform.__class__ == MovingPlatform:
        platform:Platform.state.state_vector = (request.form.getlist('platform['+str(platform_index)+'].state_vector[]'))
        platform_min_range:Platform.state.state_vector = (request.form.getlist('platform['+str(platform_index)+'].state_vector_min_range[]'))
        platform_max_range:Platform.state.state_vector = (request.form.getlist('platform['+str(platform_index)+'].state_vector_max_range[]'))
        platform_step:Platform.state.state_vector = (request.form.getlist('platform['+str(platform_index)+'].state_step[]'))
        
        #platform:MultiTransitionMovingPlatform.transition_models[0]
        print("vector"+str(platform_index))
        print(platform)

        # print(platform_index)
        # dict = platform.__dict__.keys()
        # if len(dict)>0:
        #     for key in dict:
        #         print(key)
        #         generate_detector(platform.__dict__[key],platform)


    #   if hasattr(tracker,'__dict__'):
    #     dict = tracker.__dict__.keys()
    #     if len(dict)>0:
    #         for key in dict:
    #             #print (key)
    #            # print(tracker.__dict__[key])
    #             generate_initiator(tracker.__dict__[key],tracker)
    #             print(tracker.__class__)

    #     else:
    #         print("set")            
    #         print(tracker.__class__)
    #   else:
    #     print("set no dict")
    #     if(type(tracker) is list):
    #         for el in tracker:
    #             generate_initiator(el,tracker)
    #     print(tracker.__class__) 
    
def generate_detector(detector,parent):
    if hasattr(detector,'__dict__'):
        dict = detector.__dict__.keys()
        if len(dict)>0:
            for key in dict:
               # generate_detector(el.__dict__[key])
                print(key)
        
        else:
            print("set")
            print(parent.__class__)            
            print(detector.__class__)
    else:
        if(type(detector) is list):
            for el in detector:
                print("DEEP LIST")
                generate_detector(el,parent)
        print("set no dict")
        print(parent.__class__)            
        print(detector.__class__) 



def generate_initiator(tracker,parent):

    if hasattr(tracker,'__dict__'):
        dict = tracker.__dict__.keys()
        if len(dict)>0:
            for key in dict:
                #print (key)
               # print(tracker.__dict__[key])
                generate_initiator(tracker.__dict__[key],tracker)
                #print(tracker.__class__)

        else:
            print("set")
            print(parent.__class__)            
            print(tracker.__class__)
    else:
        if(type(tracker) is list):
            for el in tracker:
                generate_initiator(el,parent)
        print("set no dict")
        print(parent.__class__)            
        print(tracker.__class__)
    #   for name, obj in inspect.getmembers(stonesoup.platform.base):
    #     if inspect.isclass(obj):
    #         print(name)
        
    #     for platform in tracker.detector.platforms:
    #             if platform.__class__ == MovingPlatform:


# def import_submodules(package_name='stonesoup'):
#     package = importlib.import_module(package_name)
#     for module_finder, name, ispkg in pkgutil.walk_packages(package.__path__, f'{package_name}.'):
#         if 'tests' in name:
#             # Let's ignore tests
#             continue
#         try:
#             importlib.import_module(name)
#         except ImportError as e:
#             warnings.warn(f'{name} failed to import {e!r}')
#         else:
#             if ispkg:
#                 import_submodules(name)
# import_submodules()

if __name__== "__main__":
    app.run(debug=True)

