# Clothing classifier
## TL;DR
The goal of this project is to develop a model that classifies pictures of clothing items in one of 10 categories:
1. T_shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Streamlit app
The repo contains a script that can be used to run a streamlit app that showcases the model.
This video shows the app's functionality.

[clothing-classifier-demo.webm](https://github.com/perezsergio/clothing-classifier/assets/129288111/aa7398bb-adff-49f3-a0ff-e9c4477235ab)

To run the app locally from your computer, simply install the required python dependencies,
either using `pip` and `requirements.txt` or `conda` and `enviroment.yaml`.
Once the environment is set up correctly, execute the following command to run the app:
```bash
streamlit run src/main.py
```

## Dataset
The model was trained on the Fashion MNIST dataset from Kaggle:
https://www.kaggle.com/datasets/zalando-research/fashionmnist/data .
This dataset contains 70 000 black and white 28x28 pixel images scrapped from zalando.com,
a popular online clothing shop.
Its name was chosen by their creators with the hope that it may one day replace the 
boring old MNIST dataset as the new standard image classification dataset.

## CNN Model
The model used to classify the images is a CNN with 2 convoluted and 2 dense layers, implemented in tensorflow.
The entire development is documented in the notebooks, 
which cover the following topics:

1. Preprocessing: transforming the images to numerical features that can be fed to a model.
2. EDA: exploration of the dataset, including visualization of a few images of each class,
    and and initial estimation of human performance.
3. Batch size: computing different benchmarks to chose the optimal batch size
4. Architecture: trying out many different CNN architecture to find the one that performs the best 
    for this problem
5. Final model: extensive evaluation of the final model


Using tensorflow, you can load the model from the `models/final_model.model` directory.
```python
from tensorflow.keras.saving import load_model

model = load_model(models/final_model.model)
```
