## informations

The following images are the result of the "full process", that is:
1) create 10 dataset (with parameters in parameters_list) (train + test, val included in train)
2) save image with samples
3) train 10 unets on the 10 datasets, make 100 predictions on the 10 test datasets, compute the IoU for each
4) create 2d color graph with the IoU scores
5) apply BN adaptation from each model to each test dataset (100 new adapted models created), predict on respective test set and compute IoU
6) create 2d color graph with IoU from adapted models
7) create 2d color graph with difference between the 2 IoUs
[WIP: 8) compute wasserstein]
8) backup relevant data
9) release memory

The entire process was reiterated 8 times

epochs = 5 \
classes = 2 \
n_train = 5000 \
n_test = 500 \
early stopping: patience = 6, min_delta = 0

parameters_list = [ \
(123, 50, 133, 50), --> 10 \
(121, 50, 135, 50), --> 14 \
(117, 50, 139, 50), --> 22 \
(113, 50, 143, 50), --> 30 \
(106, 50, 150, 50), --> 44 \
(97,  50, 159, 50), --> 62 \
(83,  50, 173, 50), --> 90 \
(65,  50, 191, 50), --> 126 \
(38,  50, 218, 50), --> 180 \
(1,   50, 255, 50), --> 254 \
]

saved all models trained (before BN adaptation)

IoU computed only on 1 mask (not on background)

## Results (IoU)
0 \
 ![alt text](https://github.com/MarcoFurlan99/IoU_graph_3d/blob/master/graphs_2d/iteration0.png?raw=true)
1 \
 ![alt text](https://github.com/MarcoFurlan99/IoU_graph_3d/blob/master/graphs_2d/iteration1.png?raw=true)
2 \
 ![alt text](https://github.com/MarcoFurlan99/IoU_graph_3d/blob/master/graphs_2d/iteration2.png?raw=true)
3 \
 ![alt text](https://github.com/MarcoFurlan99/IoU_graph_3d/blob/master/graphs_2d/iteration3.png?raw=true)
4 \
 ![alt text](https://github.com/MarcoFurlan99/IoU_graph_3d/blob/master/graphs_2d/iteration4.png?raw=true)
5 \
 ![alt text](https://github.com/MarcoFurlan99/IoU_graph_3d/blob/master/graphs_2d/iteration5.png?raw=true)
6 \
 ![alt text](https://github.com/MarcoFurlan99/IoU_graph_3d/blob/master/graphs_2d/iteration6.png?raw=true)
7 \
 ![alt text](https://github.com/MarcoFurlan99/IoU_graph_3d/blob/master/graphs_2d/iteration7.png?raw=true)

## Samples

 ![alt text](https://github.com/MarcoFurlan99/IoU_graph_3d/blob/master/graphs_2d/samples.png?raw=true)
 
## Training histories
 
 0 \
  ![alt text](https://github.com/MarcoFurlan99/IoU_graph_3d/blob/master/graphs_2d/training_histories/training_history0.png?raw=true)
 1 \
  ![alt text](https://github.com/MarcoFurlan99/IoU_graph_3d/blob/master/graphs_2d/training_histories/training_history1.png?raw=true)
 2 \
  ![alt text](https://github.com/MarcoFurlan99/IoU_graph_3d/blob/master/graphs_2d/training_histories/training_history2.png?raw=true)
 3 \
  ![alt text](https://github.com/MarcoFurlan99/IoU_graph_3d/blob/master/graphs_2d/training_histories/training_history3.png?raw=true)
 4 \
  ![alt text](https://github.com/MarcoFurlan99/IoU_graph_3d/blob/master/graphs_2d/training_histories/training_history4.png?raw=true)
 5 \
  ![alt text](https://github.com/MarcoFurlan99/IoU_graph_3d/blob/master/graphs_2d/training_histories/training_history5.png?raw=true)
 6 \
  ![alt text](https://github.com/MarcoFurlan99/IoU_graph_3d/blob/master/graphs_2d/training_histories/training_history6.png?raw=true)
 7 \
  ![alt text](https://github.com/MarcoFurlan99/IoU_graph_3d/blob/master/graphs_2d/training_histories/training_history7.png?raw=true)
  
  
