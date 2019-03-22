This project explores the use different intrepreter tools with the help of open-sourced trained models. 

# Potentially helpful tools
### SHAP
[github](https://github.com/slundberg/shap)<br />
[PAPER](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)

### ML interpretability
[github](https://github.com/tensorflow/lucid)<br />
Have to learn  svelte component to understand the repository<br />
In their model Zoo:
- inceptionV1
- CaffeNet
- AlexNet
- VGG16_caffe, VGG19_caffe
- Mobilenet-V1&V2 (in model Zoo but not implemented for visualisation demo)
- ResNet V1&V2 (in model Zoo but not implemented for visualisation demo)

### Grad-Cam
Unlike CAM, which requires feature maps to directly precede softmax layers, this approach takes sum of feature maps to calculate the final heatmap<br />
- uses Lua
[github](https://github.com/ramprs/grad-cam)<br />
 - Caffe implementation for VGG-16/VGG-19/AlexNet<br />

[PAPER](https://arxiv.org/abs/1610.02391)<br />
- python implementation for VGG16 (https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py)<br />



### Layer-wise Relevance propagation
This work precedes cam and grad-cam, getting with Taylor Expansion<br />
[PAPER](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)<br />
[GITHUB](https://github.com/atulshanbhag/Layerwise-Relevance-Propagation/blob/master/vgg/lrp.py)<br />
- Python, tensorflow specifically for VGG-16/VGG-19

# Other useful but secondary priority works
Singular Vector Canonical Correlation Analysis (SVCCA), <https://arxiv.org/abs/1706.05806><br />
<https://arxiv.org/pdf/1706.07979.pdf><br />
<br />
LIME<br />
Only implemented for inception, this API is more suitable for interpreting structured data. 
[PAPER](https://arxiv.org/abs/1602.04938)<br />
[GITHUB](https://github.com/marcotcr/lime)<br />

Loss Landscape <https://arxiv.org/abs/1712.09913><br />

Feature Visualisation<https://distill.pub/2017/feature-visualization/><br />

Back Propagation <https://www.nature.com/articles/323533a0><br />

Activation Maximising <https://arxiv.org/pdf/1312.6034.pdf> - primary purpose to expose underfitting?<br />

Sensitivity Analysis <https://arxiv.org/pdf/1608.00507.pdf> - back propagation<br />
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html<br />


## Useful Networks to be tried with interpretibility implementation
- <https://keras.io/applications/#mobilenet>
- <https://github.com/michalgdak/car-recognition>
- <https://github.com/matterport/Mask_RCNN>

## INTERESTING IDEAS WHICH WE SHOULD PURSUE:
 - use lrp with translation and averaging of inverse-translated output to denoise (used in follage identification mod.)
 - use interpretability methods of different layers on inputs of new domains to identify the kind of layers we need for transfer learning.
 - plot lrp or sensitivity curve to sense how much translation the network can tolerate
 - use it to remove artefact or add artefact to all other classes
 - First train a VGG-Net and use Grad-Cam to interpret the model. Then using the weights in trained VGG-Net to train resnet and then the faster-rcnn for more reliable result. 

 ## Conclusion
 While shap, grad cam and Lime APIs can provide a picture of what traditional neural networks try to make sense of, they do not work on more useful fully connected models. The trend led by fast R-CNN, RetinaNet and Detectron, shows that machine learning in computer-vision problems should not rely on simply networks and hence current open-source interpreter APIs are not mature yet to provide a good understanding of better networks. Also current research lacks thorough understanding of the keras API to alter any of the above APIs or implement the algorithm successfully to work with maskRCNN. 

