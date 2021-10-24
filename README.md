# Generative Adversarial Set Transformers

This repository is an unofficial Tensorflow/Keras implementation of the *Generative Adversarial Set Transformer* framework from the paper: https://www.ml.informatik.tu-darmstadt.de/papers/stelzner2020ood_gast.pdf.

~[Training GAST Animation](./images/training.gif)

## Dependencies

This repository depends on the implementation of [Set Transformers](https://arxiv.org/abs/1810.00825) found in the unofficial repository: https://github.com/DLii-Research/tf-set-transformer
Note: The MNIST dataset can be found in a separate repository found here: https://github.com/DLii-Research/mnist-pointcloud

## Implementation Notes:

* While the original GAST implementation is capable of varying the number of generated points in each point cloud, this implementation is currently incapable varying set cardinalities. As a result, all of the generated point clouds contain the maximum number of points in a given set.