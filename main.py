import os
import numpy as np
import rasterio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO

ORIGINAL_DATA_PATH = 'data/original_data'
PRETRAINED_DATA_DIR = 'data/pretrained_data'

TRAIN_DIR = os.path.join(PRETRAINED_DATA_DIR, 'train')
VAL_DIR = os.path.join(PRETRAINED_DATA_DIR, 'val')
TEST_DIR = os.path.join(PRETRAINED_DATA_DIR, 'test')

classes = [
    'AnnualCrop',
    'Forest',
    'HerbaceousVegetation',
    'Highway',
    'Industrial',
    'Pasture',
    'PermanentCrop',
    'Residential',
    'River',
    'SeaLake'
]

def create_directories():
    global PRETRAINED_DATA_DIR
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(PRETRAINED_DATA_DIR, split), exist_ok=True)
        for class_name in classes:
            os.makedirs(os.path.join(PRETRAINED_DATA_DIR, split, class_name), exist_ok=True)

def collect_training_data():
    train_data = []
    for class_name in classes:
        image_paths = collect_class_data(class_name)
        train_images, _, _ = split_data(image_paths)
        for img_path in train_images:
            with rasterio.open(img_path) as src:
                img = src.read()
            reshaped_img = img.reshape(13, -1).T
            train_data.append(reshaped_img)
    return np.vstack(train_data)

def train_pca(train_data):
    pca_model = PCA(n_components=3)
    pca_model.fit(train_data)
    return pca_model

def collect_class_data(class_name):
    class_dir = os.path.join(ORIGINAL_DATA_PATH, class_name)
    images = [f for f in os.listdir(class_dir) if f.endswith('.tif')]
    return [os.path.join(class_dir, img) for img in images]

def split_data(image_paths):
    train_images, test_images = train_test_split(image_paths, test_size=0.2, random_state=42)
    val_images, test_images = train_test_split(test_images, test_size=0.5, random_state=42)
    return train_images, val_images, test_images

def apply_pca(image_path, pca_model):
    with rasterio.open(image_path) as src:
        img = src.read()
    reduced_img = pca_model.transform(img.reshape(13, -1).T).T
    return reduced_img.reshape(3, 64, 64)

def normalize_image(img):
    return (((img - img.min()) / (img.max() - img.min())) * 255).astype(np.uint8)

def save_image(img, output_dir, img_path):
    image_name = os.path.basename(img_path).replace('.tif', '.jpg')
    image_data = Image.fromarray(img.T, 'RGB')
    image_data.save(os.path.join(output_dir, image_name), format='JPEG')

def process_split(split, images, class_name, pca_model):
    split_dir = {
        'train': TRAIN_DIR,
        'val': VAL_DIR,
        'test': TEST_DIR
    }[split]
    output_dir = os.path.join(split_dir, class_name)
    for img_path in tqdm(images, desc=f"Processing {class_name} - {split}"):
        img = apply_pca(img_path, pca_model)
        img = normalize_image(img)
        save_image(img, output_dir, img_path)

def prepare_dataset():
    create_directories()
    train_data = collect_training_data()
    pca_model = train_pca(train_data)

    for class_name in classes:
        class_data = collect_class_data(class_name)
        train_images, val_images, test_images = split_data(class_data)

        process_split('train', train_images, class_name, pca_model)
        process_split('val', val_images, class_name, pca_model)
        process_split('test', test_images, class_name, pca_model)

prepare_dataset()


print("Data preparation completed!")
model = YOLO('yolo11x-cls.pt')

model.train(
    data='data/pretrained_data/',
    imgsz=64,
    task='classify',
    epochs=100,
    batch=16,
    save=True,
    device='cuda:0',
)


model = YOLO('runs/classify/train/weights/best.pt')
model.val(data='data/pretrained_data/')
