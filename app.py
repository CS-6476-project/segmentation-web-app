from flask import Flask, send_file
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
import sys

app = Flask(__name__)

print(sys.version)

url = 'https://github.com/CS-6476-project/BSDS500/blob/master/BSDS500/data/segs/deep_learning/100007.mat?raw=true'

@app.route('/')
def main():
  urllib.request.urlretrieve(url, 'a.mat')
  mat = loadmat('a.mat')
  segs = np.uint8(mat['segs'][0, 0])

  plt.imsave('test.jpg', segs)

  return send_file('test.jpg', mimetype='image/gif')

if __name__ == '__main__':
  # Threaded option to enable multiple instances for multiple user access support
  app.run(threaded=True, port=5000)