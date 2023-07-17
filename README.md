**Fusing Single Modality Transformers for VQA**

Single Modality Transformer Fusion Model is trained on the processed DAQUAR dataset. As pre-trained transformer models are used, their weights are initialized to solve
image and text-specific tasks. The Wu-Palmer similarity score code by Mateusz et al.[9] is chosen as a primary evaluation metric since it captures the semantic similarity of
strings and works well on one-word answers.

For Experiment 1, the linear fusion technique is used with the intermediate dimension of 512, dropout of 0.5, and
batch size of 32, which are considered as a decent set of parameters, since the output dimension of the transformer encoder is 768 and it is a common practice to start with a batch
size of 32. Every language transformer (BERT, ALBERT, and RoBERTa) is paired with every other image transformer
(ViT and BEiT) and trained for 5 epochs. A small learning rate of 5e-5 is chosen since the models are already pretrained and have to be only fine-tuned to the VQA task.

The results of experiment 1 shown in Table 3 suggest that RoBERTa-ViT model produces the highest Wu-Palmer
scores, with the second best being RoBERTa-BEiT. However, the performance of both vision transformers paired
with ALBERT is comparatively low. This could be because ALBERT is a light model and higher-quality textual embedding is needed for better scores. Although the model with
ViT transformer performed best, scores from BEiT pairings are better in all other cases. This difference could be because the reported results are over a single seed. Evaluating performance over a set of seeds and averaging could
prove that BEiT pairings are better since it is pre-trained in a self-supervised manner and is proven to be more robust compared to ViT. We also conducted some experiments by
varying batch size and the hidden dimension of RoBERTaViT model with a simple linear fusion layer. Validation results shown in Fig 11 and 12 (see Appendix) suggest that
lower batch size and higher intermediate dimensionality are preferred.

![alt-text-1](https://github.com/Sindhura-b/VisualQuestionAnswering_TransformerFusion/blob/main/Colab%20Notebooks/learning_curve-batch%20size.png) ![alt-text-2](https://github.com/Sindhura-b/VisualQuestionAnswering_TransformerFusion/blob/main/Colab%20Notebooks/learning_curve-hidden%20size.png)

For Experiment 2, we took the best-performing model from Experiment 1 and explored different fusion and training methods. The results of this experiment reported in Table 2 are interesting to notice that even though MFB and MFH
layers are selectively designed to capture rich multi-modal interactions, they performed poorly in comparison to multiplicative fusion and simple linear fusion techniques for the
selected model and dataset. Initially, this behavior was attributed to slow learning by MFB and MFH modules that resulted in not being able to achieve better accuracy in 5
epochs. So, epochs were increased to 10 and the results did not improve much. On increasing epochs, multiplicative fusion still performed comparatively well up to a certain extent and both models showed no improvement in validation
scores towards the end indicating that learning has stopped. This behavior could be attributed to both the structure of the fusion layer and the dataset. Multiplicative fusion is still
shallow and does not capture deeper multi-modal interactions as every element in image encoding interacts with only one other element in text encoding, MFB technique uses
sum-pooling to capture the interaction and there aren’t any direct parameters to learn multi-modal interaction. The latest VQA models used co-attention or D-depth transformer
layers[7] to place higher importance on multi-modality interaction for better performance. Also, the dataset size used is very small as it has only 16 examples per class in the
training data. Images in DAQUAR also had significant clutter and extreme lighting conditions and even human evaluation studies produced only 50 accuracy. This is also reflected in the results of model inference shown in Fig. 10
Figure 4. WUP curves for 50 and 20 epochs, 1000 and 6795 dataset sizes respectively, where the models struggle to clearly understand the objects in the test image and provide accurate predictions.

We also extended experiment 2 by freezing the layers of both the transformers and fine-tuning only the fusion and classification layers. The results show that the performance is degraded compared to full fine-tuning, which
suggests that it is important to have a larger model to perform VQA task and also that earlier layers of the network form the baseline in learning the VQA task on DAQUAR
dataset, even though they are designed to only capture single-modality interactions.
