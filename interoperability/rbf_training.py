import numpy as np
from interoperability.python_interlop.wrapper.my_lib import MyModel
import os
from PIL import Image


def load_dataset(path, neg_label, pos_label):
    images = []
    labels = []
    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)
        if os.path.isdir(class_path):
            label = neg_label if class_name == 'metal' else pos_label
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize((32, 32))  # Resize to 32x32
                img_array = np.array(img).flatten() / 255.0  # Normalize
                images.append(img_array)
                labels.append(label)
    return np.array(images), np.array(labels)


base_dir = "../mini_dataset"
train_path = os.path.join(base_dir, "train")
X, Y = load_dataset(train_path, -1, 1)

input_dim = X.shape[1]
model_type = "rbf"
model = MyModel("rbf", input_dim, cluster_size=5, gamma=1.5, is_classification=True)
model.train(X, Y, 0.1, 100_000)

print("metal")
metal_pred = model._predict_value("../dataset/train/metal/metal_1025.jpg")
print(f"metal - prediction = {metal_pred}")

print("paper")
paper_pred = model._predict_value("../dataset/train/paper/paper_3044.jpg")
print(f"paper - prediction = {paper_pred}")

print("plastic")
plastic_pred = model._predict_value("../dataset/train/plastic/plastic_6064.jpg")
print(f"plastic - prediction = {plastic_pred}")