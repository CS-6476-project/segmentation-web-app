from flask import Flask, render_template, request
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import random
import threading
import os
from glob import glob

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

ROOT_DIR = '.'
STATIC_DIR = os.path.join(ROOT_DIR, 'static')

BASE_URL = 'https://github.com/CS-6476-project/BSDS500/'
ORIGINAL_URL = 'https://raw.githubusercontent.com/CS-6476-project/BSDS500/master/BSDS500/data/images/test/'
GROUND_TRUTH_URL = BASE_URL + 'blob/master/BSDS500/data/groundTruth/test/'
SEGS_URL = BASE_URL + 'blob/master/BSDS500/data/segs'

FEATURE_SPACES = ['hsv', 'hsv_pos', 'rgb', 'rgb_pos']
FEATURE_SPACE_NAMES = ['HSV', 'HSV + Pos', 'RGB', 'RGB + Pos']


class Algo():
  def __init__(self, url, withFeatureSpace=False):
    self.name = url.replace('_', ' ').title()
    if (withFeatureSpace):
      self.paths = ["%s/%s" % (url, x) for x in FEATURE_SPACES]
      self.path_names = ["%s, %s space" % (self.name, x) for x in FEATURE_SPACE_NAMES]
    else:
      self.paths = ["%s" % url]
      self.path_names = ["%s approach" % self.name]


ALGOS = [Algo('k_means', True), Algo('mean_shift', True), Algo('normalized_cut', True), Algo('deep_learning')]


def getGroundTruth(ground_truth_url, chosen_file_name, template_data):
  file_path = os.path.join(ROOT_DIR, 'ground_truth.mat')
  image_path = os.path.join(STATIC_DIR, '%s_ground_truth.png' % chosen_file_name)

  urllib.request.urlretrieve(ground_truth_url, file_path)
  mat = loadmat(file_path)
  ground_truth_data = mat['groundTruth']
  to_pick = np.random.randint(ground_truth_data.shape[1])
  ground_truth_data = ground_truth_data[0, to_pick][0, 0]
  ground_truth_data = np.uint8(ground_truth_data[0])
  num_segs = np.unique(ground_truth_data).size
  plt.imsave(image_path, ground_truth_data)

  template_data.append([image_path, 'Ground Truth Segmentation', num_segs])


def callback(endpoint, chosen_file_name, template_data):
  file_path = os.path.join(ROOT_DIR, '%s.mat' % endpoint[1])
  image_path = os.path.join(STATIC_DIR, '%s_%s.png' % (chosen_file_name, endpoint[1]))

  urllib.request.urlretrieve(endpoint[0], file_path)
  mat = loadmat(file_path)
  segs = np.uint8(mat['segs'][0, 0])
  num_segs = np.unique(segs).size
  plt.imsave(image_path, segs)

  template_data.append([image_path, endpoint[1], num_segs])


@app.route('/')
def main():

  query = request.args.get('q')
  invalid_query = False
  chosen_file = random.choice(FILE_NAMES)

  if query:
    if f'{query}.mat' in FILE_NAMES:
      chosen_file = f'{query}.mat'
    else:
      invalid_query = True

  endpoints = []
  for algo in ALGOS:
    for path, path_name in zip(algo.paths, algo.path_names):
      full_path = "%s/%s/%s?raw=true" % (SEGS_URL, path, chosen_file)
      endpoints.append((full_path, path_name))

  if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
  else:
    for file in glob(os.path.join(STATIC_DIR, "*.png")):
      os.remove(file)

  chosen_file_name = chosen_file.split('.')[0]
  ground_truth = "%s/%s?raw=true" % (GROUND_TRUTH_URL, chosen_file)
  template_data = []

  threads = [threading.Thread(target=callback, args=(endpoint, chosen_file_name, template_data)) for endpoint in endpoints] + [threading.Thread(target=getGroundTruth, args=(ground_truth, chosen_file_name, template_data))]

  for thread in threads:
    thread.start()

  for thread in threads:
    thread.join()

  template_data.sort(key=lambda x: x[1])

  original = '%s/%s.jpg' % (ORIGINAL_URL, chosen_file_name)
  template_data.insert(0, [original, "Original image", None])

  return render_template('main.html', template_data=template_data, image_number=chosen_file_name, invalid_query=invalid_query)


if __name__ == '__main__':
  app.run(threaded=True, port=5000)
