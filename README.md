# MentalBlackboard : Evaluating Spatial Visualization Via Mathematical Transformations

[![Paper](https://img.shields.io/badge/Paper-MentalBlackboard-blue)](https://arxiv.org/abs/2602.19357) 
[![Homepage](https://img.shields.io/badge/Website-MentalBlackboard-green)](https://mentalblackboard.github.io/) 
[![Huggingface](https://img.shields.io/badge/Benchmark-MentalBlackboard-orange)](https://huggingface.co/datasets/nlylmz/MentalBlackboard)

[Nilay Yilmaz](https://www.linkedin.com/in/nilay-yilmaz/) | Maitreya Patel | Naga Sai Abhiram Kusumba | Yixuan He | Yezhou Yang 

---
Spatial visualization is the mental ability to imagine, transform, and manipulate the spatial charac-
teristics of objects and actions. This intelligence is a part of human cognition where actions and
perception are connected on a mental level. Although it plays a significant role in mathematical thinking, its eval-
uation in state-of-the-art vision-language models (VLMs) remains relatively underexplored.
In response to this challenge, we introduce MentalBlackboard: an open-ended spatial visualization benchmark that
employs Paper Folding Test (PFT) within two core tasks: prediction and planning. Prediction integrates PFT to rotation transformations
and implements a folding-unfolding strategy for the solution. The planning task aims to interpret the final unfolded paper and determine the folds and
initial holes to reach the identical result. Both tasks require visual perception to understand the concept, visuospatial
working memory to track multiple folds and punches while processing, sequential reasoning to understand the ordered
folds/unfolds, and spatial visualization to imagine and manipulate objects. 

---

## Prediction
Predict the final hole configuration after unfolding, given the folding sequence and initial hole properties.

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/0e552c59-910e-4e89-b1b3-e58d521a6457" width="350" /><br/>
      <sub><b>Prediction — Problem</b></sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/5e421ec7-a6ab-471d-922a-2db90612a794" width="350" /><br/>
      <sub><b>Prediction — Solution</b></sub>
    </td>
  </tr>
</table>


## 📢 News  
- 🚀 [02/20/2026] The using instructions are added! 
- 🚀 [02/20/2026] We upload our MentalBlackboard benchmark to Huggingface.  

---

## 💡 Highlights  
- 🔥 **Multiple Tasks with 3D Dimension**: MentalBlackboard utilizes spatial visualization PFT tasks in an interactive 3D environment, incorporating animated objects to support two distinct problems: planning and prediction.
- 🔥 **Open-ended Evaluation**: Departing from iother patial visualization tasks rely on multiple-choice formats, which reveals only accuracy and does not elucidate the reasons for incorrect response, MentalBlackboard applies open-ended evaluation with the multi-step approach by comparing the results with ground truth values at each step.
- 🔥 **Automated Pipeline**: Unlike static datasets, the dynamic structure of the MentalBlackboard creation process enables a large-scale dataset of over 12,000 unique configurations without implementing rotations, offering a scalable and adaptable evaluation platform. 
- 🔥 **Rule-Based Transformations**: MentalBlackboard contains symmetry and rotation transformations that define rule-governed spatial manipulations and support systematic reasoning processes.
---

##  MentalBlackboard Benchmark

MentalBlackboard is built with an automated, physically grounded pipeline based on a 3D paper-folding and hole-punching simulation (VPython). Folding, rotation, punching, and unfolding are executed under strict physical constraints to avoid deformation or self-intersection, and the paper is discretized into triangular regions to support diagonal and multi-stage folds.

<img src="https://github.com/user-attachments/assets/14665ada-a2d3-45c6-b05f-db19c46b4807" width="800" />

## Results
We evaluate state-of-the-art Vision-Language Models in a zero-shot setting across video-, image-, and text-based prediction tasks and 2D planning tasks. Performance is measured using Exact Match and a custom Partial Accuracy metric designed for open-ended spatial predictions. The prediction task evaluation results are provided in the table below.

![Screenshot 2026-02-18 144330](https://github.com/user-attachments/assets/16542c93-f3da-415e-9a86-0dad80c7f814)



## 🔧 Usage 

### Installation
```bash
conda create --name mbb python=3.10
conda activate mbb

git clone git@github.com:nlylmz/mentalblackboard.git
cd mentalblackboard
pip install -r requirements.txt
```
### Benchmark

The MentalBlackboard dataset is hosted on HuggingFace. You can access the benchmark prediction and planning task data using the following codes.
```bash
from datasets import load_dataset
mbb_planning = load_dataset("nlylmz/MentalBlackboard", "planning")
mbb_prediction = load_dataset("nlylmz/MentalBlackboard", "prediction")
print(mbb_prediction["train"][0])
print(mbb_planning["train"][0])
```

### Dataset Generation

#### Prediction and Planning
To generate MentalBlackboard prediction and planning questions, run animate.py, which supports dataset creation for both tasks. Since prediction is the reverse process of planning, a single unified format is sufficient to generate data for both. You'll need to select the structure_group (how many folds and whether to include rotation), then enter the number of cases you'd like to generate. **The structure configuration includes one_step, two_step, three_step, four_step, two_step_with_rotation, three_step_with_rotation, four_step_with_rotation, five_step_with_rotation, and six_step_with_rotation.** 

After the run completes, three folders will be created to store the outputs: **folding_frames**, **unfolding_frames**, and **prediction_results**, which contain the folding frames, unfolding frames, and final result images, respectively. The corresponding metadata will be saved in **MentalBlackboard_Prediction_Data.json**.

**Note: The system captures frames by taking screenshots. Please ensure that only the animation is visible on the screen and avoid using or switching to other applications during the process.**

Run the script using the following command:
```bash
python animate.py prediction <structure_group> <count>
```

#### Create Video
To create the video from the generated folding and unfolding frames, run the command below:
```bash
python animate.py create_video
```
Two folders will be created to store the output videos: **folding_videos** and **unfolding_videos**.

#### Image - Text Mapping
To generate the image and text representations of the task, run the following command and provide the metadata JSON file as a parameter:

```bash
python img_text_map.py <json_file_path>
```
Both image and text results will be saved under the **img_text_outputs** folder.

### Model Evaluation

To evaluate models on MentalBlackboard, we provide the inference_test.py script, which downloads the MentalBlackboard dataset from Hugging Face and performs evaluation for the specified model. The model processes the input data, generates outputs, and saves the results in a JSON file. **Supported models for multimodal inputs: Qwen/Qwen2.5-VL-7B-Instruct, llava-hf/llava-onevision-qwen2-0.5b-ov-hf, and google/gemma-3-4b-it** 

#### Prediction
Run the script using the following command:
```bash
python inference_MentalBlackboard_prediction.py --model_name <huggingface_pretrained_model>
```
#### Planning
Run the script using the following command:
```bash
python inference_MentalBlackboard_planning.py --model_name <huggingface_pretrained_model>
```
Since planning tasks may have multiple valid solutions, we validate the model’s output by replaying it in the animation and verifying that it produces the correct final result. You need to enter the model's output as input_file parameter. The animation output will be saved to a file named **model_name_planning_results.json**. To run the animation for the planning task, follow the command: 
```bash
python animate.py planning <input_file>
```
After running the animation, the model’s output must be compared against the ground-truth metadata. To perform the comparison, run the following command:
```bash
python planning_scoring.py <ground_truth_json> <model_result_json>
```
Example: python planning_scoring.py PFHP_Planning_Data.json Qwen_Qwen2.5-VL-7B-Instruct_planning_results.json

## 🖋️ Citation  

```
@misc{yilmaz2026mentalblackboardevaluatingspatialvisualization,
      title={MentalBlackboard: Evaluating Spatial Visualization via Mathematical Transformations}, 
      author={Nilay Yilmaz and Maitreya Patel and Naga Sai Abhiram Kusumba and Yixuan He and Yezhou Yang},
      year={2026},
      eprint={2602.19357},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.19357}, 
}
```

