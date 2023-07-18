**Fusing Single Modality Transformers for VQA**

**Approach:**
In this approach, I explored the use of existing vision and language transformer models to build a multimodal
transformer that can perform the task of visual question answering. The pre-trained transformer architectures selected
for obtaining language embeddings are BERT, RoBERTa and AlBERT and image embeddings are ViT and BEiT. The
pre-trained models are obtained using APIs in the transformer package provided by huggingface. The problem
of solving VQA is posed as a multi-class classification problem, where the entire vocabulary of answers in the dataset
is treated as labels. 

The vision and language transformers are fused using the late fusion technique and tuned for the task. The late fusion
method is chosen because it often gives better performance because errors from multiple models are dealt with independently and hence the errors are uncorrelated. Late fusion
layers have less complex modality interaction compared to early fusion techniques and hence are easier to train comparatively. I also explored various ways to perform late
fusion and the ones chosen for implementation and comparative analysis include:

**Linear Fusion**: This involves the concatenation of the image
and textual features and passing them through a linear layer
followed by ReLU activation and dropout to generate an
intermediate output.
**Multiplicative Fusion**: This involves element-wise multiplication of image and textual features to generate an intermediate output. This method has no learnable parameters
for multi-modality interaction (fusion layer) if the image
and textual features are of the same size.
**Factorized Bi-Linear Pooling (MFB)**: Bilinear pooling supposedly captures richer pairwise interactions among
multi-modal features[4]. It also does not pose any restriction on the dimensionality of image and text features. 
**Factorized Higher-Order Pooling (MFH)**: This involves simply cascading multiple MFB modules to capture
more complex high-order interactions between multi-modal features[5]. For this project, I chose to use two MFB
modules for the implementation of MFH. After the fusion layer, a classifier with a fully connected
layer whose output dimensionality is equivalent to that of the answer space of the chosen dataset is added to the
model. As a part of this model exploration, a comparative analysis of models with transformer and fusion layer variants is performed. 

**Experiments and Results:**
Single Modality Transformer Fusion Model is trained on the processed DAQUAR dataset. As pre-trained transformer models are used, their weights are initialized to solve
image and text-specific tasks. The Wu-Palmer similarity score code by Mateusz et al.[1] is chosen as a primary evaluation metric since it captures the semantic similarity of
strings and works ill on one-word answers. Models are implemented in jupyter notebbok and training is performed in colab on NVIDIA Tesla T4 and NVIDIA Tesla P100 GPUs.

For Experiment 1, the linear fusion technique is used with the intermediate dimension of 512, dropout of 0.5, and
batch size of 32, which are considered as a decent set of parameters, since the output dimension of the transformer encoder is 768 and it is a common practice to start with a batch
size of 32. Every language transformer (BERT, ALBERT, and RoBERTa) is paired with every other image transformer
(ViT and BEiT) and trained for 5 epochs. A small learning rate of 5e-5 is chosen since the models are already pretrained and have to be only fine-tuned to the VQA task.

The results of experiment 1 shown in Table 3 suggest that RoBERTa-ViT model produces the highest Wu-Palmer
scores, with the second best being RoBERTa-BEiT. However, the performance of both vision transformers paired
with ALBERT is comparatively low. This could be because ALBERT is a light model and higher-quality textual embedding is needed for better scores. Although the model with
ViT transformer performed best, scores from BEiT pairings are better in all other cases. This difference could be because the reported results are over a single seed. Evaluating performance over a set of seeds and averaging could
prove that BEiT pairings are better since it is pre-trained in a self-supervised manner and is proven to be more robust compared to ViT. I also conducted some experiments by
varying batch size and the hidden dimension of RoBERTa-ViT model with a simple linear fusion layer. Validation results shown in Fig 11 and 12 (see Appendix) suggest that
lower batch size and higher intermediate dimensionality are preferred.

<p float="left">
  <img src="https://github.com/Sindhura-b/VisualQuestionAnswering_TransformerFusion/blob/main/Colab%20Notebooks/learning_curve-batch%20size.png" width="40%" /> <img src="https://github.com/Sindhura-b/VisualQuestionAnswering_TransformerFusion/blob/main/Colab%20Notebooks/learning_curve-hidden%20size.png" width="40%" /> 
</p>

For Experiment 2, I took the best-performing model from Experiment 1 and explored different fusion and training methods. The results of this experiment reported in Table 2 are interesting to notice that even though MFB and MFH
layers are selectively designed to capture rich multi-modal interactions, they performed poorly in comparison to multiplicative fusion and simple linear fusion techniques for the
selected model and dataset. Initially, this behavior was attributed to slow learning by MFB and MFH modules that resulted in not being able to achieve better accuracy in 5
epochs. So, epochs are increased to 10 and the results did not improve much. On increasing epochs, multiplicative fusion still performed comparatively ill up to a certain extent and both models shoid no improvement in validation
scores towards the end indicating that learning has stopped. This behavior could be attributed to both the structure of the fusion layer and the dataset. Multiplicative fusion is still
shallow and does not capture deeper multi-modal interactions as every element in image encoding interacts with only one other element in text encoding, MFB technique uses
sum-pooling to capture the interaction and there aren’t any direct parameters to learn multi-modal interaction. The latest VQA models used co-attention or D-depth transformer
layers[7] to place higher importance on multi-modality interaction for better performance. Also, the dataset size used is very small as it has only 16 examples per class in the
training data. Images in DAQUAR also had significant clutter and extreme lighting conditions and even human evaluation studies produced only 50 accuracy. This is also reflected in the results of model inference shown in Fig. 10
Figure 4. WUP curves for 50 and 20 epochs, 1000 and 6795 dataset sizes respectively, where the models struggle to clearly understand the objects in the test image and provide accurate predictions.

<p float="left">
  <img src="https://github.com/Sindhura-b/VisualQuestionAnswering_TransformerFusion/blob/main/Colab%20Notebooks/results.png" width="100%" /> 
</p>

i also extended experiment 2 by freezing the layers of both the transformers and fine-tuning only the fusion and classification layers. The results show that the performance is degraded compared to full fine-tuning, which
suggests that it is important to have a larger model to perform VQA task and also that earlier layers of the network form the baseline in learning the VQA task on DAQUAR
dataset, even though they are designed to only capture single-modality interactions.

<p float="left">
  <img src="https://github.com/Sindhura-b/VisualQuestionAnswering_TransformerFusion/blob/main/Colab%20Notebooks/inference%20results.png" width="100%" /> 
</p>

**ViLT multimodal transformer fine-tuning**

**Approach**: Vision and Language Transformer (ViLT) uses the simplest architecture for the task of visual question answering. In ViLT, the image embeddings are created in a similar fashion to textual embeddings instead of using convolutional neural networks. Both the embeddings are summed with their corresponding modal-type embedding vectors and are concatenated into a combined sequence which is updated iteratively through D-depth transformer layers [6]. Unlike many vision and language processing models, ViLT creates convolution-free embedding and uses a bigger model for modality interaction. The single modality transformer fusion model discussed above is built differently and uses a simpler modality interaction model. Hence, it was decided to compare and analyze the performance of these two models on the DAQAR dataset.  

The DAQAR dataset is converted to VQA space[2], which is then remapped to ViLT input space structure using 'ViLTProcessor' transformer class provided with the ViLT source code [8]. The dataset thus obtained is passed through a pre-trained ViLT model with a randomly initialized head. ViLT is finetuned to DAQAR dataset using the similar setup for VQA2 dataset implemented in the jupyter notebook [7].

**Experiments and Results**: 

**References**

1. Mateusz Malinowski. vqa-playground-pytorch.
https://github.com/bupt-cist/
vqa-playground-pytorch/blob/master/
calculate_wups.py, 2015. 4

2. tezansahu. Visual question answsering with transformers. https://github.com/tezansahu/
VQA-With-Multimodal-Transformers/
blob/main/notebooks/
VisualQuestionAnsiringWithTransformers.
ipynb. 3

3. https://medium.com/@nithinraok_/visual-question-ansiring-attention-and-fusion-based-approaches-ebef62fa55aa

4. Zhou Yu, Jun Yu, Jianping Fan, and Dacheng Tao. Multimodal factorized bilinear pooling with co-attention learning
for visual question ansiring, 2017.

5. Zhou Yu, Jun Yu, Chenchao Xiang, and Dacheng Tao. Beyond bilinear: Generalized multimodal factorized high-order
pooling for visual question ansiring, 2017. 

6. @misc{kim2021vilt,
      title={ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision}, 
      author={Wonjae Kim and Bokyung Son and Ildoo Kim},
      year={2021},
      eprint={2102.03334},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}

7. [https://www.google.com/search?q=vilt+finetuning&oq=vilt+finetuning&aqs=chrome.0.69i59j0i13i512j0i5i13i30j0i390i650l2j69i60j69i61j69i60.3188j0j1&sourceid=chrome&ie=UTF-8](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/ViLT/Fine_tuning_ViLT_for_VQA.ipynb)https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/ViLT/Fine_tuning_ViLT_for_VQA.ipynb

8. https://github.com/dandelin/ViLT
