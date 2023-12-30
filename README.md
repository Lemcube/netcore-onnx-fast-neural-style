
# netcore onnx fast neural style
[![Build Status](#)](link-to-build-status) ![.NET Version](#) ![License](#)

A .NET Core demo for ONNX model: Image style transformation. 

## Resources
- ONNX Models: [GitHub](https://github.com/onnx/models/tree/main/validated/vision/style_transfer/fast_neural_style)
- Original PyTorch Models: [GitHub](https://github.com/pytorch/examples/tree/main/fast_neural_style#models)
- Original Paper (2016): [Stanford University](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)

## Application Output

| Original Image |
| -------------- |
| ![Original](/result/lion.jpeg) |
| Mosaic Style | Pointilism Style | Udnie Style |
| ------------ | ---------------- | ----------- |
| ![Mosaic Style](/result/lion_mosaic.jpg) | ![Pointilism Style](/result/lion_pointilism.jpg) | ![Udnie Style](/result/lion_udnie.jpg) |

The application transforms images using various style models. Below are some examples:

### Original Image
![Original](/results/lion.jpg)

### Mosaic Style
![Mosaic Style](/results/lion_mosaic.jpg)

### Pointilism Style
![Pointilism Style](/results/lion_pointilism.jpg)

### Udnie Style
![Udnie Style](/results/lion_udnie.jpg)
