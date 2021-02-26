import os
from flask import Flask, render_template,url_for,request,redirect
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap
from stonesoup.runmanager.parameters import Parameters
from stonesoup.serialise import YAML

app = Flask(__name__)

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
        read_config_file(file)
        return render_template('config.html',res = res)
       
      #send file name as parameter to download
    return render_template('index.html')



def read_config_file(config_file): 
    config_string = config_file.read()
    tracker, ground_truth, metric_manager = YAML('safe').load(config_string)
    print(tracker)



if __name__== "__main__":
    app.run(debug=True)