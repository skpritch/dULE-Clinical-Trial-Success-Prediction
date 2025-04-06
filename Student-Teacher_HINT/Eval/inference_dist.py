import torch, os, sys
import warnings
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings("ignore")

sys.path.append('.')
from HINT.dataloader import csv_three_feature_2_dataloader
from HINT.molecule_encode import MPNN, ADMET
from HINT.icdcode_encode import GRAM, build_icdcode2ancestor_dict
from HINT.protocol_encode import Protocol_Embedding
from HINT.model import HINTModel

device = torch.device("cpu")  # or "cuda:0" if you have a GPU

def validate_phase(iter, base_name):
    """
    Loads and evaluates HINT on a specific phase (e.g. 'phase_I', 'phase_II', 'phase_III').
    """

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    datafolder = os.path.join(BASE_DIR, "data")

    # Construct file paths
    """
    train_file = os.path.join(datafolder, base_name + '_train.csv')
    valid_file = os.path.join(datafolder, base_name + '_valid.csv')
    """
    test_file  = os.path.join(datafolder, base_name + '_test.csv')

    # Load test data
    """
    train_loader = csv_three_feature_2_dataloader(train_file, shuffle=True, batch_size=30)
    valid_loader = csv_three_feature_2_dataloader(valid_file, shuffle=False, batch_size=30)
    """
    test_loader  = csv_three_feature_2_dataloader(test_file, shuffle=False, batch_size=30)

    """
    # Load pre-trained ADMET model
    admet_model_path = os.path.join(BASE_DIR, "save_model", "admet_model.ckpt")
    if not os.path.exists(admet_model_path):
        raise FileNotFoundError(f"Missing ADMET checkpoint at {admet_model_path}. Aborting.")

    admet_model = torch.load(admet_model_path, map_location=device)
    admet_model.set_device(device)

    # Load pre-trained disease and protocol encoders - maybe what is differing inference results?
    icdcode2ancestor_dict = build_icdcode2ancestor_dict()

    gram_model = GRAM(embedding_dim=50, icdcode2ancestor=icdcode2ancestor_dict, device=device)
    protocol_model = Protocol_Embedding(output_dim=50, highway_num=3, device=device)
    """

    # Load pre-trained HINT model
    hint_model_path = os.path.join(BASE_DIR, "save_model", base_name + ".ckpt")
    if not os.path.exists(hint_model_path):
        raise FileNotFoundError(f"[ERROR] No pre-trained checkpoint found for {base_name}. Aborting.")

    print(f"Loading pre-trained HINT model for {base_name} from {hint_model_path}...")
    model = torch.load(hint_model_path, map_location=device)

    # Evaluate on test set
    print(f"\n[INFO] Starting inference on {base_name} test set...")
    return model.bootstrap_test(test_loader) #iter

def main():
    """
    Loop through the three phases and run 100 iterations of validation for each.
    """
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for phase in ["phase_I", "phase_II", "phase_III"]:
        phase_scores = []
        for i in range(0, 100):
            print("="*70)
            print(f"VALIDATING {phase.upper()} - ITERATION {i}")
            print("="*70)
            scores = validate_phase(i, phase)
            phase_scores.append(validate_phase(i, phase)[2])
        output_filename = os.path.join(BASE_DIR, "results/orig_inference_val", f"{phase}_scores.pkl")
        pickle.dump(phase_scores, open(output_filename, "wb"))

if __name__ == "__main__":
    main()

    