from flask import Flask, render_template
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import random
import threading
import os

app = Flask(__name__)

FILE_NAMES = ['103006.mat', '160006.mat', '187058.mat', '140088.mat',
              '112056.mat', '147080.mat', '202000.mat', '209021.mat',
              '388067.mat', '51084.mat', '81066.mat', '335094.mat',
              '232076.mat', '157087.mat', '230098.mat', '365072.mat',
              '43051.mat', '161045.mat', '247012.mat', '49024.mat',
              '187099.mat', '41096.mat', '296058.mat', '145079.mat',
              '163096.mat', '35049.mat', '196027.mat', '94095.mat',
              '61034.mat', '247003.mat', '368037.mat', '15011.mat',
              '201080.mat', '206062.mat', '20069.mat', '230063.mat',
              '118031.mat', '16068.mat', '14092.mat', '103029.mat',
              '41085.mat', '87015.mat', '65084.mat', '97010.mat',
              '108036.mat', '384089.mat', '112090.mat', '107072.mat',
              '14085.mat', '385022.mat', '285022.mat', '188025.mat',
              '64061.mat', '317043.mat', '128035.mat', '80085.mat',
              '393035.mat', '120093.mat', '220003.mat', '80090.mat',
              '28083.mat', '146074.mat', '107014.mat', '388006.mat',
              '17067.mat', '179084.mat', '309040.mat', '198087.mat',
              '69022.mat', '226060.mat', '376086.mat', '78098.mat',
              '189029.mat', '159022.mat', '15062.mat', '43033.mat',
              '157032.mat', '196040.mat', '160067.mat', '279005.mat',
              '5096.mat', '259060.mat', '250087.mat', '140006.mat',
              '106047.mat', '372019.mat', '48017.mat', '246009.mat',
              '10081.mat', '36046.mat', '70011.mat', '104055.mat',
              '296028.mat', '250047.mat', '189013.mat', '253016.mat',
              '108069.mat', '183066.mat', '175083.mat', '257098.mat',
              '189006.mat', '228076.mat', '35028.mat', '217090.mat',
              '225022.mat', '388018.mat', '288024.mat', '117025.mat',
              '103078.mat', '69000.mat', '196088.mat', '41029.mat',
              '48025.mat', '267036.mat', '33044.mat', '16004.mat',
              '141048.mat', '196062.mat', '226043.mat', '281017.mat',
              '71076.mat', '159002.mat', '105027.mat', '207049.mat',
              '100007.mat', '344010.mat', '223060.mat', '118072.mat',
              '101084.mat', '384022.mat', '268048.mat', '268074.mat',
              '71099.mat', '130066.mat', '41006.mat', '69007.mat',
              '163004.mat', '156054.mat', '346016.mat', '23050.mat',
              '29030.mat', '100039.mat', '326085.mat', '207038.mat',
              '2018.mat', '249021.mat', '147077.mat', '253092.mat',
              '134067.mat', '347031.mat', '306051.mat', '223004.mat',
              '189096.mat', '123057.mat', '77062.mat', '79073.mat',
              '81090.mat', '145059.mat', '277053.mat', '226022.mat',
              '108004.mat', '306052.mat', '243095.mat', '118015.mat',
              '130014.mat', '289011.mat', '302022.mat', '335088.mat',
              '104010.mat', '217013.mat', '81095.mat', '45000.mat',
              '258089.mat', '6046.mat', '235098.mat', '107045.mat',
              '226033.mat', '176051.mat', '106005.mat', '3063.mat',
              '134049.mat', '290035.mat', '109055.mat', '164046.mat',
              '181021.mat', '70090.mat', '8068.mat', '185092.mat',
              '102062.mat', '208078.mat', '101027.mat', '206097.mat',
              '334025.mat', '168084.mat', '120003.mat', '326025.mat',
              '141012.mat', '100099.mat', '92014.mat', '238025.mat']

URL = 'https://github.com/CS-6476-project/BSDS500/blob/master/BSDS500/data/segs'


FEATURE_SPACES = ['hsv', 'hsv_pos', 'rgb', 'rgb_pos']
ALGOS_WITH_FEATURE_SPACES = ['k_means', 'mean_shift']
OTHER_ALGOS = ['deep_learning']
PATHS = ["/".join([x, y]) for x in ALGOS_WITH_FEATURE_SPACES for y in FEATURE_SPACES] + OTHER_ALGOS

URLS = ["/".join([URL, x]) for x in PATHS]


def callback(endpoint, index, image_paths):
  file_name = '%d.mat' % index
  image_name = 'static/%d.png' % index

  urllib.request.urlretrieve(endpoint, file_name)
  mat = loadmat(file_name)
  segs = np.uint8(mat['segs'][0, 0])
  plt.imsave(image_name, segs)

  image_paths.append([image_name, endpoint])


@app.route('/')
def main():
  chosen_file = random.choice(FILE_NAMES)
  endpoints = ["%s/%s?raw=true" % (x, chosen_file) for x in URLS]

  if not os.path.exists('static'):
    os.makedirs('static')

  image_paths = []

  threads = [threading.Thread(target=callback, args=(endpoint, index, image_paths)) for index, endpoint in enumerate(endpoints)]

  for thread in threads:
    thread.start()

  for thread in threads:
    thread.join()
  # pool = ThreadPool(8)
  # results = pool.map(callback, endpoints)

  # for index, endpoint in enumerate(endpoints):
  #   file_name = '%d.mat' % index
  #   image_name = 'static/%d.png' % index

  #   urllib.request.urlretrieve(endpoint, file_name)
  #   mat = loadmat(file_name)
  #   segs = np.uint8(mat['segs'][0, 0])
  #   plt.imsave(image_name, segs)

  #   image_paths.append(image_name)

  return render_template('main.html', image_paths=image_paths)


if __name__ == '__main__':
  app.run(threaded=True, port=5000)
