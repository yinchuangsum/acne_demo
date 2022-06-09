import io
import torch
import PIL
from fastai.vision import *
import torchvision.transforms as T
import numpy as np
import streamlit as st

path = "./data/"
size = 224
bs = 64

data = ImageDataBunch.from_folder(path, valid_pct=0.2, size=size, bs=bs)
learner = cnn_learner(data, models.resnet18) # metrics=[accuracy], callback_fns=ShowGraph)
learner.load('best_resnet')

labels = ['level_0', 'level_1', 'level_2', 'normal' ]

def load_model():
    return None


def load_labels():
    return []


def load_image():
    upload_file = st.file_uploader(label='Pick an image to upload')
    if upload_file is not None:
        image_data = upload_file.getvalue()
        st.image(image_data)
        return PIL.Image.open(io.BytesIO(image_data))
    else:
        return None


def predict(categories, image):
    img_tensor = T.ToTensor()(image)
    img_fastai = Image(img_tensor)
    # img = open_image(img_fastai)
    output = learner.predict(img_fastai)
    print(output)
    print(output[2])
    classIdx = np.argmax(output[2])
    print(labels[classIdx])
    st.write("This is", labels[classIdx], "skin")
    # preprocess = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # input_tensor = preprocess(image)
    # input_batch = input_tensor.unsqueeze(0)

    # with torch.no_grad():
    #     output = learner.predict(input_batch)
    #
    # probabilities = torch.nn.functional.softmax(output[0], dim=0)
    #
    # top5_prob, top5_catid = torch.topk(probabilities, 5)
    # for i in range(top5_prob.size(0)):
    #     st.write(categories[top5_catid[i]], top5_prob[i].item())


def main():
    st.title("Ance classifier")
    # model = load_model()
    categories = load_labels()
    image = load_image()
    result = st.button('predict')
    if result:
        st.write('Calculating results...')
        predict(categories, image)


if __name__ == '__main__':
    main()
