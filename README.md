# Clothing classifier
## TL;DR
The goal of this project is to develop a model that classifies pictures of clothing items in one of 10 categories:
1 -  T_shirt/top
2 - Trouser
3 - Pullover
4 - Dress
5 - Coat
6 - Sandal
7 - Shirt
8 - Sneaker
9 - Bag
10 - Ankle boot

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


## DL model
The model used to classify the images is a CNN with 2 convoluted and 2 dense layers, implemented in tensorflow.


Using tensorflow, you can load the model from the `models/final_model.model` directory.
```python
from tensorflow.keras.saving import load_model

model = load_model(models/final_model.model)
```
