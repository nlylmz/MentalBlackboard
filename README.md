# MentalBlackboard : Evaluating Spatial Visualization Via Mathematical Transformations

[![Paper](https://img.shields.io/badge/Paper-MentalBlackboard-blue)]() 
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

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/user-attachments/assets/bdbb1e96-ed76-4908-bcf3-6415f69aac64">
        <img src="https://github.com/user-attachments/assets/5fc82471-3e0b-4a1b-a065-70d39125e02f" width="400" />
      </a><br/>
      <sub><b>Prediction — Problem</b></sub>
    </td>
    <td align="center">
      <a href="https://github.com/user-attachments/assets/b133fb2a-af29-4b96-a239-67fe9a736528">
        <img src="https://github.com/user-attachments/assets/5fc82471-3e0b-4a1b-a065-70d39125e02f" width="400" />
      </a><br/>
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

##  MentalBlackboard

MentalBlackboard is built with an automated, physically grounded pipeline based on a 3D paper-folding and hole-punching simulation (VPython). Folding, rotation, punching, and unfolding are executed under strict physical constraints to avoid deformation or self-intersection, and the paper is discretized into triangular regions to support diagonal and multi-stage folds.


## 🔧 Usage 

### Benchmark

Our benchmark is hosted on HuggingFace. You can simply access the benchmark data using the following code.
```bash
mentalBlackboard = load_dataset("nlylmz/MentalBlackboard", "planning")
```
 or 
 ```bash
mentalBlackboard = load_dataset("nlylmz/MentalBlackboard", "prediction")
```
<!--
### Dataset Generation

To generate your PFT prediction questions, you need to run create_dataset.py which supports creating both training and testing datasets. You need to download the Train_Images and Test_Images folders to access the images required for generating questions.

Run the script using the following command:
```bash
python create_dataset.py --csv_output <path_to_csv> --dataset <training/testing> --count <number>
```
#### Arguments: 
```
json_output (str, required): Path to save the output CSV file.
dataset (str, required): Choose between training or testing datasets.
count (int, required): Total number of questions to generate. Ensure it is evenly divisible by the number of rules (7 or 19).
```
### Model Evaluation

To evaluate models on MentalBlackboard, we provide the inference_test.py script, which downloads the MentalBlackboard dataset from Hugging Face and performs evaluation for the specified model. The model processes the input data, generates outputs, and saves the results in a JSON file.

Run the script using the following command:
```bash
python inference_test_MentalBlackboard.py --model_name <huggingface_pretrained_model>
```

#### Arguments: 
```
model_name (str, required): Name of the pretrained model to use.
dataset_name (str, default: nlylmz/MentalBlackboard): Name of the Hugging Face dataset to load.
output_path (str, default: results.json): Path to save the output JSON file.
device_map (str, default: auto): Device mapping strategy (auto, cpu, cuda, etc.).
max_new_tokens (int, default: 2048): Maximum number of new tokens to generate.
temperature (float, default: 0.0): Controls randomness in text generation. Higher values produce more diverse outputs.
```

### Scoring

To score the results of VLMs, we provide the score_model_results.py script. This script processes inference results, compares them with ground truth data, generates evaluation outputs in JSONL format, and prints the scores for each step at the terminal. To access the ground truth data, you need to download the JSON file, which contains detailed information for each question. 

Run the script using the following command:
```bash
python score_model_result.py --json_file_name <path_to_mllm_results> --csv_file_name <path_to_csv>
```

#### Arguments: 
```
json_file_name (str, required): Path to the JSON file containing MLLM inference results.
batch_jsonl_file (str, default: "batch"): Path to save the generated JSONL file for batch processing.
```
-->
## 🖋️ Citation  

```
```

