#!/usr/bin/python3
# coding: utf-8

import flask
import requests
from werkzeug.utils import secure_filename
import tempfile
import os
import pymongo
import os.path


def authenticate_esb(host, auth):
    """
    Faz a autenticação com a plataforma e retorna uma sessão autenticada
    """
    s = requests.Session()
    user, password = auth.split(":")
    s.auth = requests.auth.HTTPBasicAuth(user, password)
    s.verify = False
    login_url = host+"esb/login"
    login = s.post(login_url)
    s.headers["X-XSRF-TOKEN"] = login.cookies.get("XSRF-TOKEN")
    return s, True


app = flask.Flask("sbkServer")

@app.route('/upload',methods=['POST'])
def upload_file():
    #print(flask.request.files)
    headers = {"Accept": "application/json"}
    session,_ = authenticate_esb(HOST,AUTHORIZATION)
    session.headers.update(headers)
    session.headers["X-XSRF-TOKEN"] = session.cookies.get("XSRF-TOKEN")

    for keys,values in flask.request.files.items():
        with tempfile.TemporaryDirectory() as tmpdirname:
            filename = tmpdirname + '/' + values.filename
            values.save(filename)
            with open(filename,'rb') as file:
                r = session.request(method='POST',
                                    url=HOST+'esb/ws/file/upload2',
                                    files={'file':file},
                                    verify=False)

    return r.content or str(values.filename + " SENT SUCCESSFULLY!")


@app.route('/upload_from_url',methods=['POST'])
def upload_file_from_url():
    #print(flask.request.files)
    headers = {"Accept": "application/json"}
    session,_ = authenticate_esb(HOST,AUTHORIZATION)
    session.headers.update(headers)
    session.headers["X-XSRF-TOKEN"] = session.cookies.get("XSRF-TOKEN")

    url_path = flask.request.args.get('url', type = str)
    req_file = requests.get(url_path)
    #print(url_path)
    #print(req_file.content)
    filename = url_path.split('/')[-1]

    with tempfile.TemporaryDirectory() as tmpdirname:
        full_path_filename = tmpdirname + '/' + filename
        #full_path_filename = filename
        print(full_path_filename)
        with open(full_path_filename,'wb') as file:
            file.write(req_file.content)

        with open(full_path_filename,'rb') as file:
            r = session.request(method='POST',
                                url=HOST+'esb/ws/file/upload2',
                                files={'file':file},
                                verify=False)

    return r.content or str(filename + " SENT SUCCESSFULLY!")

@app.route('/response',methods=['GET'])
def get_frame():
    filename = flask.request.args.get('filename', type = str)
    filename = os.path.splitext(filename)[0]
    #print(filename)

    db = connection[MONGO_DB]
    #db.authenticate(MONGO_USER,MONGO_PASS,source=MONGO_DB,mechanism='SCRAM-SHA-1')
    projection = {"_id":False,"_idFile":False,"hash":False,"dateRef":False,"cnpj":False}
    response = db[COLLECTION_NAME].find_one({"filename": filename},sort=[('_id', pymongo.DESCENDING)],projection=projection)

    return flask.jsonify(response or {})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
