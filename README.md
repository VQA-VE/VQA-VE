# Locating Visual Explanation for Video Question Answering

## New Task: Video Question Answering with Visual Explanation (VQA-VE)

We introduce a new task called **Video Question Answering with Visual Explanation (VQA-VE)**, which requires to generate natural language sentences as answers and provide visual explanations (i.e., locating relevant moment within the whole video) simultaneously. As shown below, a visual explanation can be taken as an evidence to justify if predicted answers are convincible and traceable, or as a supplementary that provide relevant information on the context of QA pairs, or even as a specification that indicates vague expressions or abstract concepts in QA pairs vividly. This task bridges two separate and typical visual tasks: video question answering and temporal localization, and also comes with its own set of challenges.

![Task](https://github.com/VQA-VE/VQA-VE/blob/master/pic/task.jpg "An overview of our task")
*VQA-VE requires to provide visual explanation for predicted answers. There are advantages for visual explanation: (Left) visual explanation can serve as an evidence to justify the correctness of answers; (Middle) visual explanation can provide supplementary information for the content of QA pairs; (Right) visua lexplanation can give clear indication to elaborate the vague expression in QA pairs.*

## New Dataset: ActivityNet-QA
To facilate research of VQA-VE, we construct a new dataset called **ActivityNet-QA** on top of ActivityNet Captions manually. Specifically, we generate temporally annotated QA pairs (each one is coupled with a start time and a end time to mark the relevant moment) by exploiting videos and temporally annotated descriptions in ActivityNet Captions. ActivityNet-QA features three characteristics:  
**Open-ended**: Activity-QA is an open-ended dataset that a natural language answer is expected to be produced word by word. The length distribution of questions and answers are shown below. Compared with other open-ended VideoQA datasets, our QA pairs are longer and share rich vocabulary diversity. The total number of words appearing in ActivityNet-QA is 4992.
![q_length](https://github.com/VQA-VE/VQA-VE/blob/master/pic/qlength.jpg "Question length distribution of Activity-QA")
![a_length](https://github.com/VQA-VE/VQA-VE/blob/master/pic/answerlength.jpg "Answer length distribution of Activity-QA")  

**Action-centric and time-centric**: Corresponding to dynamic and temporal characteristics of videos, QA pairs in Activity-QA feature action-centric and time-centric. Specifically, questions in Activity-QA are all about events or actions of people or objects. Most of our questions contain temporal adverbial such as ‘after’, ‘at first’, ‘the second time’ *etc*.
  
**Temporally Annotation**: Each QA pair in Activity-QA is coupled with a unique timestamps, representing the start and end time of the visual explanation. The time spans of the visual explanation are various relative to the overall video length as shown below.
![v_length](https://github.com/VQA-VE/VQA-VE/blob/master/pic/videolength.jpg "Visual explanation length distribution of Activity-QA")
  
More importantly, we double check QA pairs to make sure each one only matches one exact part in the whole video, and filter out QA pairs which are not action-centric and timecentric. Under such strict standards, we collect 12059 temporally annotated QA pairs for 7304 videos in total.

## New Model
Towards VQA-VE, we develop a new model of multi-task framework to generate answers and provide visual explanations simultaneously. Specifically, we design an answer prediction module that employs visual attention and semantic attention to fully fuse cross-modal feature and generate complete natural language sentences as answers. We also design a localization module to locate relevant moment with various time spans within the whole video using semantic information as guidance.  
![model](https://github.com/VQA-VE/VQA-VE/blob/master/pic/model.jpg "An overview of our model")
*The visual encoder and the question encoder as well as a GRU extract clip features and question
features. Then cross-modal features are refined by visual attention and semantic attention, and fed into multi-modal fusion module for fully fusing. Finally, a GRU and attention mechanism are used to generate answers and locate visual explanations.*

## New Metrics
VQA-VE is a compositional task that requires to generate natural language sentences as answers and locate relevant moment simultaneously. Only evaluating the answer quality or localization quality is not enough. We consider both and design two new metrics which can fairly measure the performance of VQA-VE task. One is called ‘hard metric’: when calculated IoU higher than the given threshold, we set it to 1.0, otherwise 0.0. Specifically, we compute ‘WUPS=n, IoU=m’ which means we set threshold n for WUPS and m for IoU, and ‘R@t’ means the top-t results we pick to compute the hard metric. Another is called ‘soft metric’, in which case we don’t set any threshold for IoU. Instead, we pick the calculated IoU as confidence score for predicted answer and multiply with the WUPS score to measure the whole task quality.

## Experiments

### Comparison of Visual Encoder
We consider three different visual features extracted by two kinds of visual encoder on Activity-QA to compare their effectiveness.
![vencoder](https://github.com/VQA-VE/VQA-VE/blob/master/pic/visual_encoder.jpg "Performance comparison of different visual encoders")
### Comparison of Question Encoder
We also experiment with two kinds of question encoder to compare their effectiveness on Activity-QA.
![qencoder](https://github.com/VQA-VE/VQA-VE/blob/master/pic/question_encoder.jpg "Performance comparison of different question encoders")
### Experiments on Activity-QA
We conduct experiments on Activity-QA, and report the results evaluated by soft metric, hard metric (R@1 and R@5) separately.
### Experiments on TVQA
TVQA is a large-scale multiple choice dataset that consists of 152.5K temporally annotated QA pairs from 21.8K video clips. We also conduct experiments on it, and report the results evaluated by hard metric (R@5).
### Experiments on Traditional VideoQA
We do a further experiment on the traditional VideoQA task in order to evaluate the impact of additional supervision of visual explanation on traditional VideoQA task.
### Qualitative results
![example](https://github.com/VQA-VE/VQA-VE/blob/master/pic/example.jpg "Qualitative results generated by our model on Activity-QA")
*Qualitative results generated by our model on Activity-QA. The timestamps on the dashed line denote ground truth moment, and
words in red are generated results by our model.*
