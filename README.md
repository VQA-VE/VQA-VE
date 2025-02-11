# Locating Visual Explanations for Video Question Answering

## New Task: Video Question Answering with Visual Explanations (VQA-VE)

We introduce a new task called **Video Question Answering with Visual Explanations (VQA-VE)**, which requires to generate natural language sentences as answers and provide visual explanations (i.e., locating relevant moment within the whole video) simultaneously. As shown below, a visual explanation can be taken as an evidence to justify if predicted answers are convincible and traceable, or as a supplementary that provide relevant information on the context of QA pairs, or even as a specification that indicates vague expressions or abstract concepts in QA pairs vividly. This task bridges two separate and typical visual tasks: video question answering and temporal localization, and also comes with its own set of challenges.  
![Task](https://github.com/VQA-VE/VQA-VE/blob/master/pic/tasks.jpg "An overview of our task")
*VQA-VE requires to provide visual explanations for predicted answers. There are advantages for visual explanations: (Left)
visual explanation can serve as an evidence to justify the correctness of answers; (Middle) visual explanation can provide supplementary
information for the content of QA pairs; (Right) visual explanation can give clear indication to elaborate the vague expressions in QA pairs.*

## New Dataset: Activity-QA
To facilate research of VQA-VE, we construct a new dataset called **Activity-QA** on top of ActivityNet Captions manually. Specifically, we generate temporally annotated QA pairs (each one is coupled with a start time and a end time to mark the relevant moment) by exploiting videos and temporally annotated descriptions in ActivityNet Captions. Activity-QA features three characteristics:  
**Open-ended**: Activity-QA is an open-ended dataset that natural language sentences are expected to be produced word by word as answers. The length distribution of questions and answers are shown below. Compared with other open-ended VideoQA datasets, our QA pairs are longer and share rich vocabulary diversity. The total number of words appearing in Activity-QA is 4992.  
![qalength](https://github.com/VQA-VE/VQA-VE/blob/master/pic/qalength.jpg "Question and answer length distribution of Activity-QA")

**Action-centric and time-centric**: Corresponding to dynamic and temporal characteristics of videos, QA pairs in Activity-QA feature action-centric and time-centric. Specifically, questions in Activity-QA are all about events or actions of people or objects. Most of our questions contain temporal adverbial such as ‘after’, ‘at first’, ‘the second time’ *etc*.
  
**Temporally Annotation**: Each QA pair in Activity-QA is coupled with a unique timestamps as the start and end time of the visual explanations. The time spans of the visual explanations are various relative to the overall video length as shown below.  
![v_length](https://github.com/VQA-VE/VQA-VE/blob/master/pic/videolength.jpg "Visual explanations length distribution of Activity-QA")
  
More importantly, we double check QA pairs to make sure each one only matches one exact part in the whole video, and filter out QA pairs which are not action-centric and timecentric. Under such strict standards, we collect 12059 temporally annotated QA pairs for 7304 videos in total.

**Examples**: `Videos we choose to show here are short limited to the size constraint of GIF in github. However, most of videos in our dataset are much longer that the lengths are always beyond one minutes. Besides, most of our videos contain more than one QA pairs.`

| Videos | Temporally Annotated QA pairs |
| ---- | ---- |
|![example1](https://github.com/VQA-VE/VQA-VE/blob/master/pic/example1.gif "Example1") | **Question**: ["What is the man doing before he shows the razor to the camera?"]  **Answer**: ["He is shaving under his chin."]  **Visual Explanation**: [0, 17.09]  |
|![example2](https://github.com/VQA-VE/VQA-VE/blob/master/pic/example2.gif "Example2") | **Question**: ["What does the man do after he gets off the tractor?"]  **Answer**: ["He walks while pushing the tractor."]  **Visual Explanation**: [32.28, 39.29]  |
|![example3](https://github.com/VQA-VE/VQA-VE/blob/master/pic/example3.gif "Example3") | **Question**: ["What does the little girl do before she stands up and walks away?"]  **Answer**: ["She slides down a blue slide in a theme park."]  **Visual Explanation**: [1.93, 17.78]  |
|![example4](https://github.com/VQA-VE/VQA-VE/blob/master/pic/example4.gif "Example4") | **Question**: ["What does the man do after he puts on his eye mask?"]  **Answer**: ["He begins to solve a cube."]  **Visual Explanation**: [6.26, 16.17]  |
|![example5](https://github.com/VQA-VE/VQA-VE/blob/master/pic/example5.gif "Example5") | **Question**: ["What is the baby doing before climbing onto the sand dune?", "What does the baby do after digging the sand off of the sand dune?"]  **Answer**: ["He is digging the sand off of the sand dune using his small hands.", "He starts to try and climb onto the sand dune and he looks at the camera once."]  **Visual Explanation**: [[0, 2.77], [2.77, 8.31]] |
|![example6](https://github.com/VQA-VE/VQA-VE/blob/master/pic/example6.gif "Example6") | **Question**: ["What does the man do before the axe is stuck on the tree?", "What happened after the man lifts his ax and brings it down with force?"]  **Answer**: ["He lifts his ax and brings it down with force.", "The axe is stuck on the tree."]  **Visual Explanation**: [[0.45, 3.24], [3.58, 4.69]]  |


## New Model
Towards VQA-VE, we develop a new model of multi-task framework to generate answers and provide visual explanations simultaneously. Specifically, we design an answer prediction module that employs visual attention and semantic attention to fully fuse cross-modal feature and generate complete natural language sentences as answers. We also design a localization module to locate relevant moment with various time spans within the whole video using semantic information as guidance.    
![model](https://github.com/VQA-VE/VQA-VE/blob/master/pic/models.jpg "An overview of our model")
*The overview of our model. The visual encoder, question encoder, and GRU extract clip features and question features. Then
multi-modal fusion module refines cross-modal features by visual attention and semantic attention, and element-wise addition, elementwise multiplication and concatenation followed by a Fully Connected (FC) layer are used to fully fuse refined features. Finally, answer prediction module and localization module are used to generate answers and provide visual explanations.*

## New Metrics
VQA-VE is a compositional task that requires to generate natural language sentences as answers and locate relevant moment simultaneously. Only evaluating the answer quality or localization quality is not enough. We consider both and design two new metrics which can fairly measure the performance of VQA-VE task. One is called ‘**hard metric**’: when calculated IoU higher than the given threshold, we set it to 1.0, otherwise 0.0. Specifically, we compute ‘WUPS=n, IoU=m’ which means we set threshold n for WUPS and m for IoU, and ‘R@t’ means the top-t results we pick to compute the hard metric. Another is called ‘**soft metric**’, in which case we don’t set any threshold for IoU. Instead, we pick the calculated IoU as confidence score for predicted answer and multiply with the WUPS score to measure the whole task quality.

## Experiments

### Comparison of Visual Encoder and Question Encoder
We consider three different visual features extracted by two kinds of visual encoder on Activity-QA, and also experiment with two kinds of question encoder to compare their effectiveness on Activity-QA.  
![comparison](https://github.com/VQA-VE/VQA-VE/blob/master/pic/comparisons.jpg "Performance comparison of different visual encoders and different question encoders")
### Experiments on Activity-QA
We conduct experiments on Activity-QA, and report the results evaluated by soft metric, hard metric (R@1 and R@5).     
![Activity-QA](https://github.com/VQA-VE/VQA-VE/blob/master/pic/Activity-QA_experiment.jpg "Experiment results on Activity-QA")
### Experiments on TVQA
TVQA is a large-scale multiple choice dataset that consists of 152.5K temporally annotated QA pairs from 21.8K video clips. We also conduct experiments on it, and report the results evaluated by soft metric, hard metric (R@1 and R@5).     
![TVQA](https://github.com/VQA-VE/VQA-VE/blob/master/pic/TVQA_experiment.jpg "Experiment results on TVQA")
### Experiments on Traditional VideoQA
We do a further experiment on the traditional VideoQA task in order to evaluate the impact of additional supervision of visual explanations on traditional VideoQA task. Experiment results demonstrate that additional supervision from visual explanations can improve the performance of models on traditional VideoQA task.    
![Traditional](https://github.com/VQA-VE/VQA-VE/blob/master/pic/TraditionalQA.jpg "Experiment results on traditional VideoQA task")
### Qualitative results    
![example](https://github.com/VQA-VE/VQA-VE/blob/master/pic/examples.jpg "Qualitative results generated by our proposed model on Activity-QA.") 
*Qualitative results generated by our proposed model on Activity-QA. Timestamps above the dashed line denote ground-truth
timestamps of visual explanations, and words in red are generated results by our proposed model.*
