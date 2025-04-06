"""
Call script as python train.py <base_name> <lambda> <d>
"""

## script.py
import torch, os, sys
torch.manual_seed(0) 
sys.path.append('..')
from HINT.dataloader import csv_three_feature_2_dataloader, generate_admet_dataloader_lst, csv_three_feature_2_complete_dataloader
from HINT.molecule_encode import MPNN, ADMET 
from HINT.icdcode_encode import GRAM, build_icdcode2ancestor_dict
from HINT.protocol_encode import Protocol_Embedding, save_sentence_bert_dict_pkl
from HINT.model import Interaction, HINT_nograph, HINTModel, HINTClassroom

device = torch.device("cpu")
if not os.path.exists("figure"):
    os.makedirs("figure")

################################################
## 1. Parse input arguments (base_name, lambda, d)
################################################
if len(sys.argv) != 4:
    print("Usage: python script.py <base_name> <lambda> <d>")
    sys.exit(1)

base_name = sys.argv[1]
_lambda = float(sys.argv[2])
d = float(sys.argv[3])

################################################
## 2. Data paths (maintained as before)
################################################
datafolder = os.path.join("..", "data")
train_file = os.path.join(datafolder, base_name + '_train.csv')
valid_file = os.path.join(datafolder, base_name + '_valid.csv')
test_file = os.path.join(datafolder, base_name + '_test.csv')

################################################
## 3. (Pretrain) - No changes here
################################################
# No modifications needed for ADMET generation logic or file naming here.

################################################
## 4. Dataloader, model build, train, inference
################################################
train_loader = csv_three_feature_2_dataloader(train_file, shuffle=True, batch_size=32)
valid_loader = csv_three_feature_2_dataloader(valid_file, shuffle=False, batch_size=32)
test_loader = csv_three_feature_2_dataloader(test_file, shuffle=False, batch_size=32)

# Protocol embeddings
protocol_embedding_file = os.path.join(datafolder, "sentence2embedding.pkl")
if os.path.exists(protocol_embedding_file):
    print("Protocol embedding file found.")
else:
    print("Protocol embedding file not found, generating protocol embeddings...")
    save_sentence_bert_dict_pkl()

# Build encoders
icdcode2ancestor_dict = build_icdcode2ancestor_dict()
gram_model_student = GRAM(embedding_dim = 50, icdcode2ancestor = icdcode2ancestor_dict, device = device)
gram_model_teacher = GRAM(embedding_dim = 50, icdcode2ancestor = icdcode2ancestor_dict, device = device)
protocol_model_student = Protocol_Embedding(output_dim = 50, highway_num=3, device = device)
protocol_model_teacher = Protocol_Embedding(output_dim = 50, highway_num=3, device = device)

# Checkpoint paths
admet_model_path = os.path.join("..", "save_model", "admet_model.ckpt")

# MPNN models
mpnn_model_student = MPNN(mpnn_hidden_size = 50, mpnn_depth=3, device = device)
mpnn_model_teacher = MPNN(mpnn_hidden_size = 50, mpnn_depth=3, device = device)

# If ADMET not trained, train it
if not os.path.exists(admet_model_path):
    admet_dataloader_lst = generate_admet_dataloader_lst(batch_size=32)
    admet_trainloader_lst = [i[0] for i in admet_dataloader_lst]
    admet_testloader_lst = [i[1] for i in admet_dataloader_lst]
    admet_model_student = ADMET(
        molecule_encoder=mpnn_model_student, 
        highway_num=2, 
        device=device, 
        epoch=3, 
        lr=5e-4, 
        weight_decay=0, 
        save_name='admet_'
    )
    admet_model_student.train(admet_trainloader_lst, admet_testloader_lst)
    torch.save(admet_model_student, admet_model_path)
else:
    admet_model_student = torch.load(admet_model_path)
    admet_model_student = admet_model_student.to(device)
    admet_model_student.set_device(device)

# Same for teacher
if not os.path.exists(admet_model_path):
    admet_dataloader_lst = generate_admet_dataloader_lst(batch_size=32)
    admet_trainloader_lst = [i[0] for i in admet_dataloader_lst]
    admet_testloader_lst = [i[1] for i in admet_dataloader_lst]
    admet_model_teacher = ADMET(
        molecule_encoder=mpnn_model_student, 
        highway_num=2, 
        device=device, 
        epoch=3, 
        lr=5e-4, 
        weight_decay=0, 
        save_name='admet_'
    )
    admet_model_teacher.train(admet_trainloader_lst, admet_testloader_lst)
    torch.save(admet_model_teacher, admet_model_path)
else:
    admet_model_teacher = torch.load(admet_model_path)
    admet_model_teacher = admet_model_teacher.to(device)
    admet_model_teacher.set_device(device)

# Instantiate student & teacher
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

################################################
## NEW: Checkpoint path with base_name, lambda, d
################################################
# Directory -> ../save_model/base_name
save_base_dir = os.path.join("..", "save_model", base_name)
os.makedirs(save_base_dir, exist_ok=True)

# For minimal changes, preserve the prior naming style but now include _lambda_ and _d
student_ckpt_path = os.path.join(save_base_dir, f"{base_name}_{_lambda}_{d}_student_{num_epochs}.ckpt")
teacher_ckpt_path = os.path.join(save_base_dir, f"{base_name}_{_lambda}_{d}_teacher_{num_epochs}.ckpt")

# If not trained, train & save
if not os.path.exists(student_ckpt_path):
    # Initialize teacher with student weights
    teacher_state_dict = student.state_dict()
    teacher.load_state_dict(teacher_state_dict)
    
    student.init_pretrain(admet_model_student)
    teacher.init_pretrain(admet_model_teacher)
    
    # Create a classroom and train
    classroom = HINTClassroom(student, teacher, device, _lambda=_lambda, d=d)
    trained_student, trained_teacher = classroom.learn(
        train_loader,
        valid_loader,
        test_loader,
        base_name=base_name,
        path=save_base_dir,   # The learning function may combine "path" with its own naming
        num_epochs=num_epochs,
        lr=0.5e-3,
        weight_decay=0
    )
    torch.save(trained_student, student_ckpt_path)
    torch.save(trained_teacher, teacher_ckpt_path)
else:
    # Already trained, just load & run inference/test
    student = torch.load(student_ckpt_path)
    teacher = torch.load(teacher_ckpt_path)

    student.bootstrap_test(test_loader)
    teacher.bootstrap_test(test_loader)
