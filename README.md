Garment Worker Productivity Prediction Using Feedforward Neural Network

1. Summary

The goal of this project is to predict the productivity 0f worker, based on the following features:

- date
- quarter
- department
- day
- team
- targeted productivity
- smv
- wip
- overtime
- incentive
- idle time
- idle men
- no of style
- no of workers
- actual productivity

The dataset is obtained from here.

2. IDE and Framework

This project is created using Sypder as the main IDE. The main frameworks used in this project are Pandas, Numpy, Scikit-learn and TensorFlow Keras.

3. Methodology

3.1 Data Pipeline

The data is first loaded and preprocessed, such that unwanted features are removed. Categorical features are encoded ordinally. Then the data is split into train-validation-test sets, with a ratio of 60:20:20.

3.2 Model Pipeline

A feedforward neural network is constructed that is catered for regression problem. The structure of the model is fairly simple. Figure below shows the structure of the model.

![model](https://user-images.githubusercontent.com/95268200/176732656-647d33f5-4ce5-4a75-ae4e-b00e5589b82d.png)

The model is trained with a batch size of 64 and for 100 epochs. Early stopping is applied in this training. The training stops at epoch 24, with a training MAE of 736 and validation MAE of 535. The two figures below show the graph of the training process, indicating the convergence of model training.

![loss_graph](https://user-images.githubusercontent.com/95268200/176732761-220d7f9c-5f1a-4667-8e82-d7e274eeca4d.PNG)
![mae_graph](https://user-images.githubusercontent.com/95268200/176732767-98ef4432-5734-4dba-8380-51e2c6d918e3.PNG)

4. Results

The model are tested with test data. The evaluation result is shown in figure below.

![test_result](https://user-images.githubusercontent.com/95268200/176732845-ecc9e88c-2462-4673-999b-491a199356b6.PNG)

The model is also used to made prediction with test data. A graph of prediction vs label is plotted, as shown in the image below.

![result](https://user-images.githubusercontent.com/95268200/176732906-931d7c99-4730-4917-8086-2d970a56d3b2.png)

Based on the graph, a clear trendline of y=x can be seen, indicating the predictions are fairly similar as labels. However, several outliers can be seen in the graph.
