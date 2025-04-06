#!/usr/bin/env python3
import os, sys, torch
torch.manual_seed(0)
import warnings

# Add parent directory to path so HINT modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings("ignore")
sys.path.append('.')
from tqdm import tqdm

from HINT.dataloader import csv_three_feature_2_dataloader, generate_admet_dataloader_lst
from HINT.molecule_encode import MPNN, ADMET
from HINT.icdcode_encode import GRAM, build_icdcode2ancestor_dict
from HINT.protocol_encode import Protocol_Embedding, save_sentence_bert_dict_pkl
from HINT.model import HINTModel

def main():
    # 1. Set device
    device = torch.device("cpu")  # or "cuda:0" if GPU is available

    # 2. Prepare your train/val/test CSVs
    train_file = os.path.join("..", "data", "all_train.csv")
    valid_file = os.path.join("..", "data", "all_valid.csv")
    test_file  = os.path.join("..", "data", "all_test.csv")

    # 3. Build or load the ADMET model
    admet_ckpt = os.path.join("..", "save_model", "admet_model.ckpt")
    mpnn_model = MPNN(mpnn_hidden_size=50, mpnn_depth=3, device=device)
    
    if not os.path.exists(admet_ckpt):
        print("No ADMET checkpoint found – training ADMET from scratch.")
        admet_dataloader_lst = generate_admet_dataloader_lst(batch_size=32)
        admet_trainloaders = [dl[0] for dl in admet_dataloader_lst]
        admet_validloaders = [dl[1] for dl in admet_dataloader_lst]
        
        admet_model = ADMET(
            molecule_encoder=mpnn_model,
            highway_num=2,
            device=device,
            epoch=3,
            lr=5e-4,
            weight_decay=0,
            save_name="admet_"
        )
        admet_model.train(admet_trainloaders, admet_validloaders)
        torch.save(admet_model, admet_ckpt)
        print("ADMET model trained and saved.")
    else:
        print("Found existing ADMET checkpoint. Loading ...")
        admet_model = torch.load(admet_ckpt, map_location=device)
        admet_model = admet_model.to(device)
        admet_model.set_device(device)

    # 4. Make the DataLoaders for the main classification task
    train_loader = csv_three_feature_2_dataloader(train_file, shuffle=True, batch_size=32)
    valid_loader = csv_three_feature_2_dataloader(valid_file, shuffle=False, batch_size=32)
    test_loader  = csv_three_feature_2_dataloader(test_file, shuffle=False, batch_size=32)

    # 5. Check for and (if necessary) retrain ICD and protocol encoders
    # ICD-10 encoder: check for the ancestor dictionary file
    icd_dict_file = os.path.join("..", "data", "icdcode2ancestor_dict.pkl")
    if os.path.exists(icd_dict_file):
        print("ICD code ancestor dictionary found.")
    else:
        print("ICD code ancestor dictionary not found, building it...")
        build_icdcode2ancestor_dict()

    # Protocol embeddings: check for the sentence embedding file
    protocol_embedding_file = os.path.join("..", "data", "sentence2embedding.pkl")
    if os.path.exists(protocol_embedding_file):
        print("Protocol embedding file found.")
    else:
        print("Protocol embedding file not found, generating protocol embeddings...")
        save_sentence_bert_dict_pkl()

    # Now load the ICD dictionary
    icdcode2ancestor = build_icdcode2ancestor_dict()
    gram_model = GRAM(embedding_dim=50, icdcode2ancestor=icdcode2ancestor, device=device)
    protocol_model = Protocol_Embedding(output_dim=50, highway_num=3, device=device)

    # 6. Construct our final HINT model
    hint_model_path = os.path.join("..", "save_model", "all","all_0.9_0.0_teacher_2.ckpt")
    os.makedirs(os.path.dirname(hint_model_path), exist_ok=True)

    # If we already have a trained checkpoint, load it and run bootstrap testing only
    if os.path.exists(hint_model_path):
        print(f"Found existing HINT model checkpoint at {hint_model_path}. Loading for inference only...")
        hint_model = torch.load(hint_model_path, map_location=device)
        hint_model = hint_model.to(device)
        print("\n===== Bootstrap Test on the test set =====")
        hint_model.bootstrap_test(test_loader)
        return
    else:
        # If no checkpoint, construct the model, train, then save
        hint_model = HINTModel(
            molecule_encoder=mpnn_model,
            disease_encoder=gram_model,
            protocol_encoder=protocol_model,
            device=device,
            global_embed_size=50,
            highway_num_layer=2,
            prefix_name="my_phaseI",  # for figure naming
            gnn_hidden_size=50,
            epoch=5,  # how many epochs to train
            lr=1e-3,
            weight_decay=0
        )
        # Pre‐initialize with the ADMET pretraining
        hint_model.init_pretrain(admet_model)

        print("===== Training HINT Model =====")
        hint_model.learn(train_loader, valid_loader, test_loader)

        print(f"\nSaving trained HINT model to {hint_model_path}...")
        torch.save(hint_model, hint_model_path)

        print("\n===== Final Bootstrap Test on the test set =====")
        hint_model.bootstrap_test(test_loader)

if __name__ == "__main__":
    main()
