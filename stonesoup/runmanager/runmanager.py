import os
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

        stateVector = tracker.initiator.initiator.prior_state.state_vector
        covar = tracker.initiator.initiator.prior_state.covar
        return render_template('config.html',res = res, stateVector = stateVector, covar=covar, tracker=tracker)
       
      #send file name as parameter to download
    return render_template('index.html')



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

      runManager = initialise_tracker(runManager,tracker)
      
      runManager = initialise_deleter(runManager)
     
      return runManager



def initialise_tracker(runManager,tracker):
      if tracker.initiator.__class__== GaussianParticleInitiator:
        #NUMBER PARTICLES
        runManager.number_particles.append(request.form.getlist('number_particles')[0])
        runManager.number_particles_min_range = request.form.getlist('number_particles_min_range')[0]
        runManager.number_particles_max_range = request.form.getlist('number_particles_max_range')[0]
        runManager.number_particles_step = request.form.getlist('number_particles_step')[0]

        if tracker.initiator.initiator.__class__ == SinglePointInitiator:
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
      
      print(request.form.getlist('time_steps_since_update_max_range'))
      return runManager

    
if __name__== "__main__":
    app.run(debug=True)