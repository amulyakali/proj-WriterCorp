import os
import sys
sys.path.append(os.getcwd())
from flask import Flask, redirect, url_for, request, render_template, jsonify, send_file
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from main.start import model_extract
import cv2
import traceback
import json
import numpy as np


# Define a flask app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def convert(o):
    if isinstance(o, np.generic): return o.item()
    raise TypeError

@app.route('/icr_res', methods=['GET', 'POST'])
@cross_origin()
def upload():
    if request.method == 'POST':

        # Get the file from post request
        files = request.files
        basepath = os.path.dirname(__file__)
        file = files['file']
        slip_fp = os.path.join(
            basepath, 'uploads','jpg',secure_filename(file.filename))
        file.save(slip_fp)
        try:
            res = model_extract(slip_fp)
            if res:
                mod_path = os.path.join("mod_jpg",os.path.basename(slip_fp))
                cv2.imwrite(mod_path,res.oriented_orig_img)
                return jsonify(json.dumps({"lines":res.lines,"path":os.path.join(os.getcwd(),mod_path)},default=convert))
            return jsonify("No text extracted")
        except:
            print("EXCEPTION in model_extract")
            traceback.print_exc()
        return "Nothing extracted"

if __name__=="__main__":
    app.run(host="ec2-52-7-45-213.compute-1.amazonaws.com", debug=False, port=8103, threaded=False)

