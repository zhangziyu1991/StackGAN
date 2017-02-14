import scipy.io
import numpy as np
import pickle
import os
import pandas as pd

BIRD_DIR = 'Data/birds'

def load_filenames(data_dir):
    filepath = data_dir + '/filenames_flying.pickle'
    with open(filepath, 'rb') as f:
        filenames = pickle.load(f)
    print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
    # filenames = filenames[0:10]
    return filenames

def load_bbox(data_dir):
    bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
    df_bounding_boxes = pd.read_csv(bbox_path,
                                    delim_whitespace=True,
                                    header=None).astype(int)

    filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
    df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
    filenames = df_filenames[1].tolist()
    print('Total filenames: ', len(filenames), filenames[0])

    filename_bbox = {img_file[:-4]: [] for img_file in filenames}
    numImgs = len(filenames)
    for i in xrange(0, numImgs):
        # bbox = [x-left, y-top, width, height]
        bbox = df_bounding_boxes.iloc[i][1:].tolist()

        key = filenames[i][:-4]
        filename_bbox[key] = bbox

    return filename_bbox

def custom_crop(img, bbox):
    # bbox consists of (x-left, y-top, width, height)
    imsiz = img.shape  # [height, width, channel]
    center_x = int((2 * bbox[0] + bbox[2]) / 2)
    center_y = int((2 * bbox[1] + bbox[3]) / 2)
    R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
    y1 = np.maximum(0, center_y - R)
    y2 = np.minimum(imsiz[0], center_y + R)
    x1 = np.maximum(0, center_x - R)
    x2 = np.minimum(imsiz[1], center_x + R)
    img_cropped = img[y1:y2, x1:x2]
    return img_cropped


def convert_birds_dataset_pickle(inpath, set):
    filename_bbox = load_bbox(inpath)

    sketches = list()
    ids = list()

    train_dir = os.path.join(inpath, set)
    train_filenames = load_filenames(train_dir)
    for i in range(len(train_filenames)):
        fn = train_filenames[i]

        bbox = filename_bbox[fn]
        mat = scipy.io.loadmat('sketches_flying/{}.mat'.format(fn))
        sketch = mat['sketch']
        sketch = custom_crop(sketch, bbox)
        sketch = scipy.misc.imresize(sketch, [76, 76], 'bicubic').astype('float32')
        sketch -= 0.5

        sketches.append(sketch)
        ids.append(np.int64(fn[:3]))

    print('Saving to {}'.format('Data/birds/{}/sketches_flying.pickle'.format(set)))
    pickle.dump(sketches, open('Data/birds/{}/sketches_flying.pickle'.format(set), 'wb'))
    pickle.dump(ids, open('Data/birds/{}/class_info_flying.pickle'.format(set), 'wb'))

def generate_pickle_filenames(text_file):
    with open(text_file, "r") as f:
        filenames = f.read().splitlines()
    for i in range(len(filenames)):
        filenames[i] = filenames[i][:-4]
    pickle.dump(filenames, open('Data/birds/train/filenames_flying.pickle', 'w'))

if __name__ == '__main__':
    generate_pickle_filenames('filenames_flying_bird.txt')
    convert_birds_dataset_pickle(BIRD_DIR, 'train')
    # convert_birds_dataset_pickle(BIRD_DIR, 'test')