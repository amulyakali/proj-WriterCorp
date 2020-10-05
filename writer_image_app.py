from flask import Flask,request, send_file
from flask_cors import CORS, cross_origin
import os


basepath = os.getcwd()
host_address = "ec2-52-7-45-213.compute-1.amazonaws.com"
port = 8104

# Define a flask app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/getImage')
@cross_origin()
def sendImage():
    path = request.args.get('path')
    if os.path.exists(os.path.join(basepath,"mod_jpg",path)):
        return send_file(os.path.join(basepath,"mod_jpg",path), mimetype='image/jpg')
    else:
        return send_file(os.path.join(basepath, "uploads","jpg", path), mimetype='image/jpg')

if __name__ == '__main__':
    app.run(host=host_address,debug=False,port=port,threaded=False)