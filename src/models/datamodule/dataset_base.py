import numpy as np
import cv2
import os

from torch.utils.data import Dataset

def read_images_multiprocessing(file_paths, is_lazy=False, img_read_flag=cv2.IMREAD_GRAYSCALE, color_cvt_flag=None):
    # TODO: if data fit into ram then get_worker else lazy_memory_worker        
    def get_worker(path, i, return_list):
        img = cv2.imread(path, img_read_flag)
        if color_cvt_flag:
            img = cv2.cvtColor(img, color_cvt_flag)
        return_list.append([img, i])            

    def lazy_memory_worker(path, i, return_list):
        cv2.imread(path)
        
    import multiprocessing as mp
    manager = mp.Manager()
    jobs = []
    total_imgs = len(file_paths)        
    if is_lazy:
        worker = lazy_memory_worker
        return_list = None
    else:
        worker = get_worker 
        return_list = manager.list()
        
    print(f'total images: {total_imgs}\nstart loading...')
    
    for i, path in enumerate(file_paths):
        p = mp.Process(target=worker, args=(path, i, return_list))
        jobs.append(p)
        p.start()
        if i % 1000 == 0: 
            print(f'{i}/{total_imgs} loaded!')
            for proc in jobs:
                proc.join()
            jobs = []
    for proc in jobs:
        proc.join()  

    print(f'{total_imgs}/{total_imgs} loaded!')
      
    if is_lazy:
        return None, None
    else:                        
        return list(zip(*return_list))


def check_label_and_img_path(labels, img_paths):
    print('Checking labels and img_paths... (if nothing showed up then it\'s ok)')
    assert len(labels) > 0 and len(img_paths) > 0, 'The length of labels and img_paths should > 0'
    selected_img_paths = np.random.choice(img_paths, 10)
    for path in selected_img_paths:
        assert os.path.exists(path), f'Image path should exists: {path}'

def check_img_not_none(img, img_path):
    if img is None:
        raise ValueError(f'Image is None: {img_path}')
        
def check_label_to_img_index(label_to_img_index):
    print('Checking label_to_img_index is not None... (if nothing showed up then it\'s ok)')
    selected_labels = np.random.choice(list(label_to_img_index.keys()), 10)
    for label in selected_labels:
        assert len(label_to_img_index[label]) > 1, f'label_to_img_index should not contain 0 elem list: {label_to_img_index}'


class DatasetBase(Dataset):
    def __init__(self, cfg, transform=None, train=True, is_img_gray=False):
        self.img_paths = []
        self.images = []
        self.labels = []
        self.label_set = set()
        self.length = 0

        self.is_img_gray = is_img_gray
        self.img_read_flag = cv2.IMREAD_GRAYSCALE if self.is_img_gray else cv2.IMREAD_COLOR
        self.color_cvt_flag = None if self.is_img_gray else cv2.COLOR_BGR2RGB

        self.transform = transform
        self.train = train
        self.random_state = cfg['seed']
        self.train_or_test = 'train' if train else 'test'

    def __getitem__(self, index):
        label = self.get_label(index)
        extra_info = 'target' # use for different stage or device
        img = self._get_transformed_image(index)
        return img, label, extra_info

    def _get_transformed_image(self, idx):
        img = self.get_image(idx)             
        if self.transform is not None:
            img = self.transform(image=img)['image']
        return img


    def __len__(self):
        return self.length

    def get_image(self, index):
        if len(self.images):
            return self.images[index]
        else:
            img_path = self.img_paths[index]
            return  self._read_image(img_path)       

    def get_label(self, index):
        return self.labels[index]

    def _read_image(self, img_path):
        img = cv2.imread(img_path, self.img_read_flag)
        check_img_not_none(img, img_path)

        if self.color_cvt_flag:
            img = cv2.cvtColor(img, self.color_cvt_flag)
        return img

    def _read_all_images(self):
        total_imgs = len(self.img_paths)
        
        for i, img_path in enumerate(self.img_paths):                        
            img = self._read_image(img_path)
            self.images.append(img)
            if i % 1000 == 0: 
                print(f'{i}/{total_imgs} loaded!')
        print(f'{total_imgs}/{total_imgs} loaded!')                

    def _read_images_into_ram(self, using_multiprocessing=False):
        print('image is gray: ', self.is_img_gray, ', img read flag: ', self.img_read_flag)
        print('BGR to RGB flag: ', self.color_cvt_flag)
        print('Reading and checking images are not Noe... (if nothing showed up then it\'s ok)')
        if using_multiprocessing:
            # lazily-read in the very first time to increase the reading speed afterward
            read_images_multiprocessing(self.img_paths, is_lazy=True)
        self._read_all_images()

    def setup(self, labels, img_paths, using_multiprocessing=False, read_into_ram=True):        
        check_label_and_img_path(labels, img_paths)
        self.labels = labels
        self.img_paths = img_paths
        self.label_set = set(self.labels)
        if read_into_ram:
            self._read_images_into_ram(using_multiprocessing=using_multiprocessing)
        self.length = len(self.img_paths)        

    def get_num_classes(self):
        return len(self.label_set)

    def get_info(self):        
        info = f'{self.train_or_test} dataset length: {len(self)}, class num: {self.get_num_classes()}'
        return info        
