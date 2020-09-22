# HCET: Hierarchical Clinical Embedding with Topic Modelling on Electronic Health Record for Predicting Depression
HCET is a temopral deep learning model which leverage the data heterogeneity of EHR by building hierachical embedding for diagosis, procedure, medication, demographic information and topic features from clinical notes. It learns the inherent structure for different types of EHR data and effectivelty process temporal information, which indicates a novel approach to constuct deep learnig models on EHR.
  
#### Running HCET

**STEP 1: Installation**  

1. We used [python](https://www.python.org/) version 3.6.5, and [TensorFlow](https://www.tensorflow.org/install) version 1.14.

2. [CUDA](https://developer.nvidia.com/cuda-downloads) was also used and highly suggested for GPU computation.

3. Download/clone the HCET code  

**STEP 2: Prepare your dataset** 

Dataset for HCET is prepared through python pickle pacage and the format is in pickled list of list of list. Each list corresponds to patients, visits, and medical codes (diagnosis, procedure codes, medication, demographics and topics). First, all data sources from EHR are mapped to an integer as categorical variables. According to the hierarchical structure, a single visit is composed of a list of integers. Similarly, a patient is composed of a list of visits. For example, [[9,20,503]] means a patient has one visit with code 9, 20, and 503. A patient with two visits with code [1,2] and [3,4,5], respectively, can be converted to a list of list [[1,2], [3,4,5]]. Hence, multiple patients can be represented as [[[1,2], [3,4,5]], [[9,20,503]], [[9,2,5], [8],[33,2]]], which means there are three patients, where the first patient made two visits and the second patient made oen visit and the third patient made three visits. This list of list of list needs to be pickled using cPickle. 

The preprocess file contains function to prepare the dataset mention above. It needs:
  * Table of demographics
  * Table of topic features
  * Table of ICD-9 codes
  * Table of CPT codes
  * Table of medication
  * Table of signifcant(unique) feature for ICD-9
  * Table of signifcant(unique) feature for CPT
  * Table of signifcant(unique) feature for medication
  * Table of diagnosis time of depressed patients

**STEP 3: Run HCET code** 

Code can be excuted either in through python or Jupyter notebook.
