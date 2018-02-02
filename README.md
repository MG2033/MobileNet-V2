# MobileNet-V2
An implementation of `Google MobileNet-V2` introduced in PyTorch. According to the authors, `MobileNet-V2` improves the state of the art performance of mobile models on multiple tasks and benchmarks. Its architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers opposite to traditional residual models which use expanded representations in the input an MobileNetV2 uses lightweight depthwise convolutions to filter features in the intermediate expansion layer.

Link to the original paper: [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)

This implementation was made to be an example of a common deep learning software architecture. It's simple and designed to be very modular. All of the components needed for training and visualization are added.

## Inverted Residuals with Linear Bottlenecks
<div align="center">
<img src="https://github.com/MG2033/MobileNet-V2/blob/master/figures/irc.png"><br><br>
</div>

## Usage
### Main Dependencies
 ```
 Python 3 and above
 pytorch 0.3
 numpy 1.13.1
 tqdm 4.15.0
 bunch 1.0.1
 matplotlib 2.0.2
 tensorboardX 1.0
 ```
### Train and Test
1. Prepare your data, then create a dataloader class such as `cifar10data.py` and `cifar100data.py`.
2. Create a .json config file for your experiments. Use the given .json config files as reference.

### Run
```
python main.py config/yourjsonconfigfile.json
```

### Experiments
Due to the lack of computational power. I trained on CIFAR-10 dataset as an example to prove correctness, and was able to achieve test top1-accuracy of 90.9%.


#### Tensorboard Visualization
Tensorboard is integrated with the project using `tensorboardX` library which proved to be very useful as there is no official visualization library in pytorch.

These are the learning curves for the CIFAR-10 experiment.

<div align="center">
<img src="https://github.com/MG2033/MobileNet-V2/blob/master/figures/tb.png"><br><br>
</div>

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

