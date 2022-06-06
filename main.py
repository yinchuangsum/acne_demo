import io
import torch
from PIL import Image
from torchvision import transforms

import streamlit as st


def load_model():
    return None


def load_labels():
    return []


def load_image():
    upload_file = st.file_uploader(label='Pick an image to upload')
    if upload_file is not None:
        image_data = upload_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def predict(model, categories, image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        st.write(categories[top5_catid[i]], top5_prob[i].item())


def main():
    st.title("Ance classifier")
    load_image()


if __name__ == '__main__':
    main()
