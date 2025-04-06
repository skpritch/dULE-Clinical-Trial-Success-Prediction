#!/usr/bin/env python3
import os, sys, torch
torch.manual_seed(0)
import warnings

# Add parent directory so HINT modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings("ignore")
sys.path.append('.')
from tqdm import tqdm

from HINT.dataloader import generate_admet_dataloader_lst
from HINT.molecule_encode import MPNN, ADMET
from HINT.icdcode_encode import build_icdcode2ancestor_dict
from HINT.protocol_encode import save_sentence_bert_dict_pkl

def main():
    # 1. Set device (change to "cuda:0" if you have GPU)
    device = torch.device("cpu")
    
    # 2. ADMET Pretraining
    # ADMET uses MoleculeNet’s cooked files (which are independent of clinical trial raw_data.csv)
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
        print("ADMET model trained and saved at:")
        print(admet_ckpt)
    else:
        print("Found existing ADMET checkpoint. Loading ...")
        admet_model = torch.load(admet_ckpt, map_location=device)
        admet_model = admet_model.to(device)
        admet_model.set_device(device)
    
    # 3. Check for ICD-10 encoder file (the ICD ancestor dictionary)
    icd_dict_file = os.path.join("..", "data", "icdcode2ancestor_dict.pkl")
    if os.path.exists(icd_dict_file):
        print("ICD code ancestor dictionary found at:")
        print(icd_dict_file)
    else:
        print("ICD code ancestor dictionary not found, building it...")
        build_icdcode2ancestor_dict()  # This function builds and saves the pickle.
        print("ICD code ancestor dictionary created at:")
        print(icd_dict_file)
    
    # 4. Check for Protocol embeddings file
    protocol_embedding_file = os.path.join("..", "data", "sentence2embedding.pkl")
    if os.path.exists(protocol_embedding_file):
        print("Protocol embedding file found at:")
        print(protocol_embedding_file)
    else:
        print("Protocol embedding file not found, generating protocol embeddings...")
        save_sentence_bert_dict_pkl()
        print("Protocol embedding file generated at:")
        print(protocol_embedding_file)
    
    # 5. Final message
    print("\nAll encoder outputs are now available.")
    print("ADMET checkpoint:", admet_ckpt)
    print("ICD dictionary:", icd_dict_file)
    print("Protocol embeddings:", protocol_embedding_file)

if __name__ == "__main__":
    main()
