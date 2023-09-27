import torch
from model import transformerEncoderNet
from classifier import transformerClassifier

model = torch.load("/home/yaoshuai/tools/BGC-Prophet/modelSave/transformerEncoder_TD_loss/bS_32_dE_200_lR_0.0005_mL_128_d_320_nH_5_nEL_2_tdP_0.1_mdP_0.1_TD/transformerEncoder_Model_TD_28.pt")
torch.save(model.state_dict(), "./annotator.pt")

classifier = torch.load("./modelSave/transformerClassifier/transformerClassifier_128_5_2_0.5_0.1_0.01_150_0.1_1.0/transformerClassifier_149.pt")
torch.save(classifier.state_dict(), "./dist/classifier.pt")