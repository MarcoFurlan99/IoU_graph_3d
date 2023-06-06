## informations

epochs = 5 \
classes = 2 \
n_train = 5000 \
n_test = 500 \
early stopping: patience = 6, min_delta = 0

parameters_list = [
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
  
  
