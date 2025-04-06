## 1. import 
## 2. input & hyperparameter
## 3. pretrain 
## 4. 'dataloader, model build, train, inference'
################################################

# we are getting some absolutely heinous issues with gradients modified in-place
# the provision thesis, since in the stripped-down code our issue is only with the teacher model,
# is that the teacher model is sharing stuff with the student, so when the student does backprop,
# it modifies stuff in-place, which creates issues

## 1. import 
import torch, os, sys
torch.manual_seed(0) 
sys.path.append('..')
from HINT.dataloader import csv_three_feature_2_dataloader, generate_admet_dataloader_lst, csv_three_feature_2_complete_dataloader
from HINT.molecule_encode import MPNN, ADMET 
from HINT.icdcode_encode import GRAM, build_icdcode2ancestor_dict
from HINT.protocol_encode import Protocol_Embedding, save_sentence_bert_dict_pkl
from HINT.old_cosine_sim_model import Interaction, HINT_nograph, HINTModel, HINTClassroom
device = torch.device("cpu")
if not os.path.exists("figure"):
	os.makedirs("figure")

## 2. data
base_name = 'hint'
datafolder = os.path.join("..", "data")
train_file = os.path.join(datafolder, base_name + '_train.csv')
valid_file = os.path.join(datafolder, base_name + '_valid.csv')
test_file = os.path.join(datafolder, base_name + '_test.csv')


## 4. dataloader, model build, train, inference
train_loader = csv_three_feature_2_dataloader(train_file, shuffle=True, batch_size=32) 
valid_loader = csv_three_feature_2_dataloader(valid_file, shuffle=False, batch_size=32) 
test_loader = csv_three_feature_2_dataloader(test_file, shuffle=False, batch_size=32) 

# Protocol embeddings: check for the sentence embedding file
protocol_embedding_file = os.path.join(datafolder, "sentence2embedding.pkl")
if os.path.exists(protocol_embedding_file):
    print("Protocol embedding file found.")
else:
    print("Protocol embedding file not found, generating protocol embeddings...")
    save_sentence_bert_dict_pkl()

## build encoders
icdcode2ancestor_dict = build_icdcode2ancestor_dict()
gram_model_student = GRAM(embedding_dim = 50, icdcode2ancestor = icdcode2ancestor_dict, device = device)
gram_model_teacher = GRAM(embedding_dim = 50, icdcode2ancestor = icdcode2ancestor_dict, device = device)
protocol_model_student = Protocol_Embedding(output_dim = 50, highway_num=3, device = device)
protocol_model_teacher = Protocol_Embedding(output_dim = 50, highway_num=3, device = device)

## checkpoint paths
admet_model_path = os.path.join("..", "save_model/admet_model.ckpt")
hint_checkpoint_path = os.path.join("..", "save_model/" + base_name + "_cosine_student_teacher")

## handle ADMET
# (separately, since we'll instantiate models either way, given that we're using state dictionaries)
mpnn_model_student = MPNN(mpnn_hidden_size = 50, mpnn_depth=3, device = device)
if not os.path.exists(admet_model_path):
    admet_dataloader_lst = generate_admet_dataloader_lst(batch_size=32)
    admet_trainloader_lst = [i[0] for i in admet_dataloader_lst]
    admet_testloader_lst = [i[1] for i in admet_dataloader_lst]
    admet_model_student = ADMET(molecule_encoder = mpnn_model_student, 
                        highway_num=2, 
                        device = device, 
                        epoch=3, 
                        lr=5e-4, 
                        weight_decay=0, 
                        save_name = 'admet_')
    admet_model_student.train(admet_trainloader_lst, admet_testloader_lst)
    torch.save(admet_model_student, admet_model_path)
else:
    admet_model_student = torch.load(admet_model_path)
    admet_model_student = admet_model_student.to(device)
    admet_model_student.set_device(device)

mpnn_model_teacher = MPNN(mpnn_hidden_size = 50, mpnn_depth=3, device = device)
if not os.path.exists(admet_model_path):
    admet_dataloader_lst = generate_admet_dataloader_lst(batch_size=32)
    admet_trainloader_lst = [i[0] for i in admet_dataloader_lst]
    admet_testloader_lst = [i[1] for i in admet_dataloader_lst]
    admet_model_teacher = ADMET(molecule_encoder = mpnn_model_student, 
                        highway_num=2, 
                        device = device, 
                        epoch=3, 
                        lr=5e-4, 
                        weight_decay=0, 
                        save_name = 'admet_')
    admet_model_teacher.train(admet_trainloader_lst, admet_testloader_lst)
    torch.save(admet_model_teacher, admet_model_path)
else:
    admet_model_teacher = torch.load(admet_model_path)
    admet_model_teacher = admet_model_teacher.to(device)
    admet_model_teacher.set_device(device)

## instantiate student & teacher models
# (necessary since we load our state_dictionaries into instantiated classes)
num_epochs = 10
student = HINTModel(
     molecule_encoder=mpnn_model_student,
     disease_encoder=gram_model_student,
     protocol_encoder=protocol_model_student,
     device=device,
     global_embed_size=50,
     highway_num_layer=2,
     prefix_name=base_name,
     gnn_hidden_size=50,
     epoch=num_epochs,
     lr=1e-3,
     weight_decay=0
)
teacher = HINTModel(
     molecule_encoder=mpnn_model_teacher,
     disease_encoder=gram_model_teacher,
     protocol_encoder=protocol_model_teacher,
     device=device,
     global_embed_size=50,
     highway_num_layer=2,
     prefix_name=base_name+"_teacher",
     gnn_hidden_size=50,
     epoch=num_epochs,
     lr=1e-3,
     weight_decay=0
)

if not os.path.exists(hint_checkpoint_path + f"_student_{num_epochs}.ckpt"):
	
    ## initialize teacher with student weights
	# (seems reasonable, consistent with pretraining setting)
    teacher_state_dict = student.state_dict()
    teacher.load_state_dict(teacher_state_dict)
	
    student.init_pretrain(admet_model_student)
    teacher.init_pretrain(admet_model_teacher)
    
    # create a classroom and train
    classroom = HINTClassroom(student, teacher, device, _lambda=0.9, d=0)
    trained_student, trained_teacher = classroom.learn(
		train_loader, valid_loader, test_loader, path=hint_checkpoint_path, num_epochs=num_epochs, lr=0.5e-3, weight_decay=0
    )
	
else:
	
    student, teacher = torch.load(hint_checkpoint_path + f"_student_{num_epochs}.ckpt"), torch.load(hint_checkpoint_path + f"_student_{num_epochs}.ckpt")

    student.bootstrap_test(test_loader)
    teacher.bootstrap_test(test_loader)


"""
PR-AUC   mean: 0.5645 
F1       mean: 0.6619 
ROC-AUC  mean: 0.5760 
"""