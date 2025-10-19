import wandb
import argparse

parser = argparse.ArgumentParser(description="Upload NPZ motion file to WandB Registry")
parser.add_argument("--npz_file", type=str, required=True, help="Path to the NPZ motion file")
parser.add_argument("--collection_name", type=str, required=True, help="Name for the collection in WandB Registry")
args = parser.parse_args()

REGISTRY_NAME = "motions"
COLLECTION_NAME = args.collection_name

run = wandb.init(project="csv_to_npz", name=COLLECTION_NAME)

logged_artifact = run.log_artifact(artifact_or_path=args.npz_file, name=COLLECTION_NAME, type=REGISTRY_NAME)

run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}")
