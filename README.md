# Locating Visual Explanation for Video Question Answering

## New Task: Video Question Answering with Visual Explanation (VQA-VE)

We introduce a new task called **Video Question Answering with Visual Explanation (VQA-VE)**, which requires to generate natural language sentences as answers and provide visual explanations (i.e., locating relevant moment within the whole video) simultaneously. As shown in Figure 1, a visual explanation can be taken as an evidence to justify if predicted answers are convincible and traceable, or as a supplementary that provide relevant information on the context of QA pairs, or even as a specification that indicates vague expressions or abstract concepts in QA pairs vividly. This task bridges two separate and typical visual tasks: video question answering and temporal localization, and also comes with its own set of challenges.

![Task](https://github.com/VQA-VE/VQA-VE/blob/master/pic/task.jpg "Task")
*Figure 1. VQA-VE requires to provide visual explanation for predicted answers. There are advantages for visual explanation: (Left) visual explanation can serve as an evidence to justify the correctness of answers; (Middle) visual explanation can provide supplementary information for the content of QA pairs; (Right) visua lexplanation can give clear indication to elaborate the vague expression in QA pairs.*

## New Dataset: ActivityNet-QA
To facilate research of VQA-VE, we construct a new dataset called **ActivityNet-QA** on top of ActivityNet Captions manually. Specifically, we generate temporally annotated QA pairs (each one is coupled with a start time and a end time to mark the relevant moment) by exploiting videos and temporally annotated descriptions in ActivityNet Captions. ActivityNet-QA features three characteristics:  
**Open-ended**: Activity-QA is an open-ended dataset that a natural language answer is expected to be produced word by word. The length distribution of questions and answers are shown in Figure 2 and Figure 3. Compared with other open-ended VideoQA datasets, our QA pairs are longer and share rich vocabulary diversity. The total number of words appearing in ActivityNet-QA is 4992.
<center>![q_length](https://github.com/VQA-VE/VQA-VE/blob/master/pic/qlength.jpg "Question length distribution")</center>
<center>*Figure 2. Question length distribution of Activity-QA.*</center>
<center>![a_length](https://github.com/VQA-VE/VQA-VE/blob/master/pic/answerlength.jpg "Answer length distribution")</center>
<center>*Figure 3. Answer length distribution of Activity-QA.*</center>
  
**Action-centric and time-centric**: Corresponding to dynamic and temporal characteristics of videos, QA pairs in Activity-QA feature action-centric and time-centric. Specifically, questions in Activity-QA are all about events or actions of people or objects. Most of our questions contain temporal adverbial such as ‘after’, ‘at first’, ‘the second time’ *etc*.
  
**Temporally Annotation**: Each QA pair in Activity-QA is coupled with a unique timestamps, representing the start and end time of the visual explanation. The time spans of the visual explanation are various relative to the overall video length as shown in Figure 4.
<center>![v_length](https://github.com/VQA-VE/VQA-VE/blob/master/pic/videolength.jpg "Visual explanation length distribution")</center>
<center>*Figure 4. Visual explanation length distribution of Activity-QA.*</center>
  
More importantly, we double check QA pairs to make sure each one only matches one exact part in the whole video, and filter out QA pairs which are not action-centric and timecentric. Under such strict standards, we collect 12059 temporally annotated QA pairs for 7304 videos in total.

## New Model


## New Metrics

## Experiments
