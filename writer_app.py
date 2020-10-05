import os
import sys
sys.path.append(os.getcwd())
from flask import Flask, redirect, url_for, request, render_template, jsonify, send_file
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from main.start import *
import cv2
import traceback
import json
import numpy as np


# Define a flask app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

basepath = os.getcwd()
host_address = "ec2-52-7-45-213.compute-1.amazonaws.com"
port = 8103
img_port = 8104

def convert(o):
    if isinstance(o, np.generic): return o.item()
    raise TypeError

@app.route('/getImage')
@cross_origin()
def sendImage():
    path = request.args.get('path')
    if os.path.exists(os.path.join(basepath,"mod_jpg",path)):
        return send_file(os.path.join(basepath,"mod_jpg",path), mimetype='image/jpg')
    else:
        return send_file(os.path.join(basepath, "uploads","jpg", path), mimetype='image/jpg')

@app.route('/icr_res', methods=['GET', 'POST'])
@cross_origin()
def upload():
    if request.method == 'POST':
        print("Inside ICR_RES")

        # Get the file from post request
        files = request.files
        basepath = os.path.dirname(__file__)
        file = files['file']
        slip_fp = os.path.join(
            basepath, 'uploads','jpg',secure_filename(file.filename))
        file.save(slip_fp)
        print("saved ",slip_fp)
        try:
            ocr_op = model_extract(slip_fp)
            if ocr_op:
                res = ocr_op["ocr_res"]
                mod_path = os.path.join("mod_jpg",os.path.basename(slip_fp))
                # cv2.imwrite(mod_path,res.oriented_orig_img)
                return jsonify(json.dumps({"lines":res.lines,"path":os.path.join(os.getcwd(),mod_path)},default=convert))
            return jsonify("No text extracted")
        except:
            print("EXCEPTION in model_extract")
            traceback.print_exc()
        return "Nothing extracted"

@app.route('/writer_extract', methods=['GET', 'POST'])
@cross_origin()
def extract_fields():
    if request.method == 'POST':

        # Get the file from post request
        files = request.files
        basepath = os.path.dirname(__file__)
        file = files['file']
        slip_fp = os.path.join(
            basepath, 'uploads','jpg',secure_filename(file.filename))
        file.save(slip_fp)
        try:
            ocr_op = model_extract(slip_fp)
            if ocr_op:
                [vals, slipType] = fields_extract(ocr_op)
            return jsonify(createResponse(vals, ocr_op["ocr_res"], slipType))
        except:
            print("EXCEPTION in model_extract")
            traceback.print_exc()
        return jsonify(createErrorResponse(slip_fp))

if __name__=="__main__":
    app.run(host=host_address, debug=False, port=port, threaded=False)

