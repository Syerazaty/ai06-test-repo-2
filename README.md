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





The model is trained with a batch size of 64 and for 100 epochs. Early stopping is applied in this training. The training stops at epoch 37, with a training MAE of 736 and validation MAE of 535. The two figures below show the graph of the training process, indicating the convergence of model training.

![loss_graph](https://user-images.githubusercontent.com/95268200/176451948-f329da46-7173-4943-aa04-c079f9857c96.PNG)

![mae_graph](https://user-images.githubusercontent.com/95268200/176452085-5bb8840f-3dcc-47c7-a5c5-c62e7fc18502.PNG)

4. Results

The model are tested with test data. The evaluation result is shown in figure below.

![test_result](https://user-images.githubusercontent.com/95268200/176452181-02a01d66-5c8c-4495-9d8d-8dd425d26e9e.PNG)

The model is also used to made prediction with test data. A graph of prediction vs label is plotted, as shown in the image below.


![result](https://user-images.githubusercontent.com/95268200/176452135-1793f140-3365-482b-a914-dbbbfcbe68ca.png)

Based on the graph, a clear trendline of y=x can be seen, indicating the predictions are fairly similar as labels. However, several outliers can be seen in the graph.
