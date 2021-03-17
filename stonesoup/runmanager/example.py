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
from stonesoup.runmanager.utils import generate_covar, generate_random_array, generate_random_covar, generate_random_int
from stonesoup.initiator.simple import GaussianParticleInitiator,SinglePointInitiator
from stonesoup.platform.base import MovingPlatform, MultiTransitionMovingPlatform, Platform
from stonesoup.base import Base
import stonesoup
from stonesoup.tracker.simple import SingleTargetTracker
from stonesoup.initiator.base import Initiator


def initialise_detector(detector,  parentList):
  if hasattr(detector, "properties"):
    properties = detector.properties
    if len(properties)>0:
      for property in properties:    
        parentList.append(detector.__name__)
        initialise_detector(getattr(detector,property),parentList)
        parentList.remove(detector.__name__)
        #print(property)
      # print(getattr(detector,property))
    else: 
      parentList.append(detector.__class__.__name__)
      #print("WITH PROPERTIES")
      #print(detector.__class__.__name__)
      #parentList.remove(detector.__class__.__name__)

  else:
      if(detector is list):
        for el in detector:
          parentList.append(detector.__name__)
          initialise_detector(el,parentList)
          parentList.remove(detector.__name__)

      else: 
        print("WITHOUT")
        parentList.append(detector.__name__)
        print(parentList)
        parentList.remove(detector.__name__)



var = Initiator

print(type(var))

if hasattr(var,"subclasses"):

    subclasses = var.subclasses
    print(subclasses)
    if len(subclasses)>0:
        for s_class in subclasses:
            print("PRIOR CLASS NAME "+str(var))
            print("CLASS NAME -- "+str(s_class.__name__))
            properties= s_class.properties
            for property in properties:
                print(property)
                print(s_class.properties[property].cls)
    else:
        print("CLASS "+str(var))
        properties= var.properties
        for property in properties:
            print(var.properties[property].cls)
# else:
#     print("print")

#     print("CLASS NAME "+str(var))
#     if hasattr(var, "properties"):
#         properties= var.properties
#         for property in properties:
#             print(property)
    

#initialise_detector(Tracker,[])