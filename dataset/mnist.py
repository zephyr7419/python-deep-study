import urllib.request
import os.path
import gzip
import pickle
import os
import numpy as np

# TensorFlow의 스토리지에서 MNIST 데이터셋 다운로드
url_base = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'

key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = os.path.join(dataset_dir, "mnist.pkl")

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name):
    file_path = os.path.join(dataset_dir, file_name)

    if os.path.exists(file_path):
        return

    print(f"{file_name} 다운로드 중... {url_base}")
    try:
        urllib.request.urlretrieve(url_base + file_name, file_path)
        print("완료")
    except Exception as e:
        print(f"{file_name} 다운로드 실패. 에러: {e}")


def download_mnist():
    for file_name in key_file.values():
        _download(file_name)


def _load_label(file_name):
    file_path = os.path.join(dataset_dir, file_name)

    print(f"{file_name}를 NumPy 배열로 변환 중...")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("완료")

    return labels


def _load_img(file_name):
    file_path = os.path.join(dataset_dir, file_name)

    print(f"{file_name}를 NumPy 배열로 변환 중...")
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("완료")

    return data


def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset


def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("피클 파일 생성 중...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("완료!")


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == '__main__':
    init_mnist()
