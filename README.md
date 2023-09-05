# Multi-Class Image Classification using ResNet50

# 1. Project Overview

The Image Classification of Five Flower Classes project aims to build a machine learning model capable of classifying images of flowers into one of the five predefined classes: **Rose**, **Tulip**, **Sunflower**, **Daisy**, and **Dandelion**. The primary goal is to create a reliable system that can automatically identify and categorize different types of flowers based on input images. This project has applications in botany, horticulture, and image recognition tasks.

# 2. Project Structure and Dependencies

## 2.1. Structure

- `requirements.txt`
    - Project dependencies.
- `source_code`
    - `resnet_prg.ipynb`
        - Code for **ResNet50**.
    - `Model.ipynb`
        - Code for other models.
- `main.py`
    - This is flask implementation for the project.
- `resnet.pkl`
    - This is the model saved as a pickle file.

## 2.2. Dependencies

Ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Matplotlib and Seaborn
- Scikit-learn
- OpenCV
- Jupyter Notebook

Use the following command to install the dependencies.

```bash
pip install -r requirements.txt
```

# 3. Dataset Description

- Link for the dataset is given below:
    
    [Flower Classification](https://www.kaggle.com/datasets/sauravagarwal/flower-classification)
    
- Specific details like number of images per class for train, test, and validation are shown using a Tableau Dashboard. Click the following link to access it.
    
    [](https://public.tableau.com/app/profile/vedant.nimbalkar/viz/Project_16935462142580/ImageClassificationDashboard?publish=yes)
    
- The dataset used is the one that is already partitioned.

# 4. Data Augmentation

- Data augmentation is a crucial technique in image classification tasks, especially when dealing with limited training data.
- It involves applying various transformations to the original images to create new training samples.
- This process not only increases the size of the training dataset but also helps improve the model's generalization by exposing it to different variations of the same data.
- In this project, data augmentation was applied using the **`ImageDataGenerator`** class provided by the `Keras` library.
- The **`ImageDataGenerator`** allows for on-the-fly data augmentation, ensuring that each batch of images provided to the model during training is slightly different, reducing overfitting and improving the model's ability to generalize to unseen data.
- The following augmentations were used:
    - **Rotation**: Randomly rotates the image by a certain degree.
    - **Width and Height Shift**: Shifts the width and height of the image by a fraction of its total width and height.
    - **Shear**: Applies shear mapping to the image.
    - **Zoom**: Randomly zooms into or out of the image.
    - **Horizontal Flip**: Flips the image horizontally.

# 5. Model Architecture

The image classification model in this project is built using a modified `ResNet-50` architecture, which is a deep convolutional neural network (CNN) pre-trained on the `ImageNet` dataset. This section outlines the key components and layers of the model architecture.

## ResNet50 Base Model

- The base of our model is ResNet-50, which is loaded with pre-trained weights from the ImageNet dataset. The input shape of the model is set to (224, 224, 3), indicating that it accepts color images with dimensions 224x224 pixels.
- To prevent the pre-trained ResNet-50 layers from being updated during training and to retain their valuable features, we freeze all layers in the base model.
- On top of the ResNet-50 base, we add custom classification layers to adapt the model to our specific task of flower classification.
    - **Global Average Pooling Layer**: Global Average Pooling reduces the spatial dimensions of the feature maps while retaining important information. It's a form of dimensionality reduction.
    - **Dense Layers with Dropout**: We follow the Global Average Pooling layer with several fully connected Dense layers, each with a ReLU activation function to capture complex patterns in the data. Dropout layers with a dropout rate of 0.5 are added after each Dense layer to reduce overfitting.
    - **Output Layer**: The final output layer consists of a Dense layer with a `softmax` activation function, producing five output classes, one for each flower category.

# 6. Training and Evaluation

- Before training, we compiled the model with the following configurations:
    - **Optimizer:** We used the `Adam` optimizer, which is a popular choice for training deep neural networks.
    - **Loss Function:** For this multi-class classification task, we used the `sparse categorical cross-entropy` loss function, which is well-suited for optimizing models that output probability distributions over multiple classes.
- We used a batch size of 32, meaning that the model was updated after processing each batch of 32 images.
- The training process was set to run for 10 epochs, which means that the entire training dataset was processed 10 times.
- We used the test dataset generated by the **`test_generator`** to evaluate the model's performance on previously unseen images.

# 7. Results

The results section provides an overview of the performance and outcomes of our image classification project, including the model's accuracy, visualizations, and any additional insights gained from the experiment.

![Untitled](Multi-Class%20Image%20Classification%20using%20ResNet50%20d37c09a9eddf45bda78871c0ca1ea084/Untitled.png)

![Untitled](Multi-Class%20Image%20Classification%20using%20ResNet50%20d37c09a9eddf45bda78871c0ca1ea084/Untitled%201.png)

![Untitled](Multi-Class%20Image%20Classification%20using%20ResNet50%20d37c09a9eddf45bda78871c0ca1ea084/Untitled%202.png)

# 8. Deployment
