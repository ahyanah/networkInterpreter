This project explores the use different intrepreter tools with the help of open-sourced trained models. 

# Potentially helpful tools
### SHAP
[github](https://github.com/slundberg/shap)
[PAPER](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)

### ML interpretability
[github](https://github.com/tensorflow/lucid)

### Grad Cam
[github](https://github.com/ramprs/grad-cam)
[PAPER](https://arxiv.org/abs/1610.02391)

### Layer-wise Relevance propagation
[PAPER](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)

# Other useful but secondary priority works
Singular Vector Canonical Correlation Analysis (SVCCA), <https://arxiv.org/abs/1706.05806>
<https://arxiv.org/pdf/1706.07979.pdf>
LIME <https://arxiv.org/abs/1602.04938>
Loss Landscape <https://arxiv.org/abs/1712.09913>
Feature Visualisation<https://distill.pub/2017/feature-visualization/>
Back Propagation <https://www.nature.com/articles/323533a0>
Activation Maximising <https://arxiv.org/pdf/1312.6034.pdf> - primary purpose to expose underfitting?
Sensitivity Analysis <https://arxiv.org/pdf/1608.00507.pdf> - back propagation
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html


# Useful Networks to be tried with interpretibility implementation
- <https://keras.io/applications/#mobilenet>
- <https://github.com/michalgdak/car-recognition>
- <https://github.com/matterport/Mask_RCNN>

# INTERESTING IDEAS WHICH WE SHOULD PURSUE:
 - use lrp with translation and averaging of inverse-translated output to denoise (used in follage identification mod.)
 - use interpretability methods of different layers on inputs of new domains to identify the kind of layers we need for transfer learning.
 - plot lrp or sensitivity curve to sense how much translation the network can tolerate
 - use it to remove artefact or add artefact to all other classes