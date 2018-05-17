# resnet-network-slimming

This is an implementation of the algorithm in "Learning Efficient Convolutional Networks through Network Slimming".

I implemented the resnet18 network pruning. For each residual block, I only cropped the first layer. This ensures that the network after pruning is the same as the original network structure. Otherwise, shortcuts need to be convolved to match the dimensions.
