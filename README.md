# Classify_white_blood_cell_types
Given an image of a white blood cell, classifies the type as neutrophil, eosinophil, monocyte or lymphocyte

data: https://www.kaggle.com/paultimothymooney/blood-cells
1. python3 train.py
2. python3 classify.py <path to saved model><path to image>
   prints the white blood cell type
   
Sample command:
  python3 classify.py ../model/13.model.epoch ../download.jpeg
   
prints:
  Loading model
  Model Loaded
  MONOCYTE
