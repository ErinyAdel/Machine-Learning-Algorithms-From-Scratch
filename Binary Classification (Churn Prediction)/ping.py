# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 13:43:03 2021
@author: Eriny
"""

from flask import Flask

app = Flask("ping")

@app.route('/ping', methods=['GET'])
def ping():
    return "PONG"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
    
## curl http://0.0.0.0:9696/ping
## curl http://0.0.0.0:9696/ping