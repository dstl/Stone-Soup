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

app = Flask(__name__)


runManager = RunManager()

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

    
        file = request.files['configFile']
        # if user does not select file, browser also
        # submit a empty part without filename

       # res.num_runs = request.form('num_runs')
        #res.num_processes = request.form('num_processes')
        #res.output = request.form('filename')

        if file.filename == '':
            print('no filename')
            return render_template('index.html')
        else:
            filename = secure_filename(file.filename)
    #        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("saved file successfully")
        
        res = Parameters()
        res.filename = file.filename
        res.num_runs = request.form['num_runs']
        res.num_processes = request.form['num_processes']
        res.output = request.form['filename']     
        tracker, ground_truth, metric_manager = read_config_file(file)

        stateVector = tracker.initiator.initiator.prior_state.state_vector
        print(type(tracker.detector.platforms[0].transition_model.model_list[0]))
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
        runManager = get_data()
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

        print(runManager.number_particles)

        return "OK"


def get_data():
      runManager = RunManager()
    
      runManager = initialise_tracker(runManager)
    
     
      return runManager



def initialise_tracker(runManager):
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

      #NUMBER PARTICLES
      runManager.number_particles.append(request.form.getlist('number_particles')[0])
      runManager.number_particles_min_range = request.form.getlist('number_particles_min_range')[0]
      runManager.number_particles_max_range = request.form.getlist('number_particles_max_range')[0]
      runManager.number_particles_step = request.form.getlist('number_particles_step')[0]

      return runManager



def generate_covar(covarList,length):
    covar = np.zeros((length,length))

    for i in range (0,length):
        for j in range(0,length):
            covar[i,j] = covarList[i*length+j]
    
    return covar


def generate_random_array(min_value,max_value, n_values):
    array = np.zeros((n_values))
    actualValue = min_value
    for i in range(0,n_values):
        if(min_value[i]<max_value[i]):
            array[i]=(randint(min_value[i],max_value[i]))
        else:
             array[i]=0

    return array


def generate_random_covar(min_value,max_value,n_values):
    matrix=np.zeros((n_values,n_values))

    for i in range(0,n_values):
        for j in range(0,n_values):
            if(min_value[0][i,j]<max_value[0][i,j]):
                matrix[i,j]=randint(min_value[0][i,j],max_value[0][i,j])
    
    return matrix



def generate_random_int(min_value,max_value):
    value = 0
    if min_value<max_value:
        value=randint(min_value,max_value)

    return value 
    
if __name__== "__main__":
    app.run(debug=True)