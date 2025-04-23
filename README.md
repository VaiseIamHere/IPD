# MedMamba For DR
Improving MedMamba Vision Model, for better classification of Diabetic Retinopathy.

# Paste to run on kaggle:
!git clone https://github.com/VaiseIamHere/IPD.git
%cd IPD
!pip install -r requirements.txt

<!-- Train -->
!python train.py <activationoption> <batch_size> <num_workers>

<!-- Test -->
!python test.py <activationoption>
