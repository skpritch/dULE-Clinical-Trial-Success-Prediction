#!/usr/bin/env python3
import torch, os, sys
torch.manual_seed(0)
sys.path.append('..')

# Import your modules here, plus the COSINE-based model
from HINT.dataloader import csv_three_feature_2_dataloader, generate_admet_dataloader_lst
from HINT.molecule_encode import MPNN, ADMET 
from HINT.icdcode_encode import GRAM, build_icdcode2ancestor_dict
from HINT.protocol_encode import Protocol_Embedding, save_sentence_bert_dict_pkl
# IMPORTANT: We now import from our new Cosine-based file
from HINT.cosine_sim_model import HINTModel, HINTClassroom

device = torch.device("cpu")
if not os.path.exists("figure"):
    os.makedirs("figure")

################################################
## 1. Parse input arguments (base_name, lambda, d)
################################################
if len(sys.argv) != 4:
    print("Usage: python cosine_train.py <base_name> <lambda> <d>")
    sys.exit(1)

base_name = sys.argv[1]
_lambda = float(sys.argv[2])
d = float(sys.argv[3])

################################################
## 2. Data paths
################################################
datafolder = os.path.join("..", "data")
train_file = os.path.join(datafolder, base_name + '_train.csv')
valid_file = os.path.join(datafolder, base_name + '_valid.csv')
test_file  = os.path.join(datafolder, base_name + '_test.csv')

################################################
## 3. Pretrain ADMET (no changes except reading from known path)
################################################
admet_model_path = os.path.join("..", "save_model", "admet_model.ckpt")

################################################
## 4. Dataloaders, Model Build, Train, Inference
################################################
train_loader = csv_three_feature_2_dataloader(train_file, shuffle=True, batch_size=32)
valid_loader = csv_three_feature_2_dataloader(valid_file, shuffle=False, batch_size=32)
test_loader  = csv_three_feature_2_dataloader(test_file, shuffle=False, batch_size=32)

# Protocol embeddings
protocol_embedding_file = os.path.join(datafolder, "sentence2embedding.pkl")
if os.path.exists(protocol_embedding_file):
    print("Protocol embedding file found.")
else:
    print("Protocol embedding file not found, generating protocol embeddings...")
    save_sentence_bert_dict_pkl()

# Build encoders
icdcode2ancestor_dict = build_icdcode2ancestor_dict()
gram_model_student = GRAM(embedding_dim=50, icdcode2ancestor=icdcode2ancestor_dict, device=device)
gram_model_teacher = GRAM(embedding_dim=50, icdcode2ancestor=icdcode2ancestor_dict, device=device)
protocol_model_student = Protocol_Embedding(output_dim=50, highway_num=3, device=device)
protocol_model_teacher = Protocol_Embedding(output_dim=50, highway_num=3, device=device)

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

# Instantiate the final HINT models
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

# Directory -> ../save_model/cos_<base_name>
save_base_dir = os.path.join("..", "save_model", "cos_" + base_name)
os.makedirs(save_base_dir, exist_ok=True)

# Checkpoint paths
student_ckpt = os.path.join(save_base_dir, f"cos_{base_name}_{_lambda}_{d}_student_{num_epochs}.ckpt")
teacher_ckpt = os.path.join(save_base_dir, f"cos_{base_name}_{_lambda}_{d}_teacher_{num_epochs}.ckpt")

# If not trained, do so
if not os.path.exists(student_ckpt):
    # Initialize teacher = student
    teacher_state_dict = student.state_dict()
    teacher.load_state_dict(teacher_state_dict)

    # Pre‐init from ADMET
    student.init_pretrain(admet_model_student)
    teacher.init_pretrain(admet_model_teacher)

    # Classroom w/ COS penalty
    classroom = HINTClassroom(student, teacher, device, _lambda=_lambda, d=d)

    trained_student, trained_teacher = classroom.learn(
        train_loader,
        valid_loader,
        test_loader,
        base_name=base_name,         # we'll parse & pass it
        path=save_base_dir,
        num_epochs=num_epochs,
        lr=5e-4,                     # or 0.5e-3 if you prefer
        weight_decay=0
    )

    torch.save(trained_student, student_ckpt)
    torch.save(trained_teacher, teacher_ckpt)
    print(f"Saved cos checkpoints to {save_base_dir}")
else:
    # Already trained
    student = torch.load(student_ckpt)
    teacher = torch.load(teacher_ckpt)

    student.bootstrap_test(test_loader)
    teacher.bootstrap_test(test_loader)
