from pathlib import Path
import torch
from torch.utils.data import DataLoader

from data.components.smiles_odor_dataset import SmilesOdorDataset

def test():
    csv_file = Path("data/smiles_odor_dataset.csv")
    print(f"--- 1. Vérification du fichier CSV : {csv_file.resolve()} ---")
    assert csv_file.exists(), f"Le fichier {csv_file} est introuvable !"

    print("\n--- 2. Chargement du dataset ---")
    dataset = SmilesOdorDataset(data_dir=csv_file)
    print(f"Nombre total de molécules chargées : {len(dataset)}")
    print(f"Nombre de classes d'odeurs : {len(dataset.label_names)}")

    print("\n--- 3. Inspection du premier échantillon (index 0) ---")
    sample_data, sample_target = dataset[0]
    print(f"SMILES : {sample_data['smiles']}")
    print(f"Positions 3D shape : {sample_data['pos'].shape} (type: {sample_data['pos'].dtype})")
    print(f"Numéros atomiques shape : {sample_data['atomic_nums'].shape} ({sample_data['atomic_nums']})")
    print(f"Target shape : {sample_target.shape} (somme des odeurs positives: {sample_target.sum().item()})")

    print("\n--- 4. Test d'un DataLoader (batch_size=1) ---")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for batch_data, batch_target in loader:
        print(f"Batch data 'pos' shape : {batch_data['pos'].shape}")
        print(f"Batch target shape : {batch_target.shape}")
        break

    print("\n Succès : le dataset fonctionne correctement !")

if __name__ == "__main__":
    test()
