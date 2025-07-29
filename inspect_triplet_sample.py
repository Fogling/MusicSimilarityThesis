import os
import torch

# Path to one subgenre folder in your preprocessed data
SUBGENRE_PATH = "preprocessed_features/Chill House"  # change to match your folder

def unwrap_and_print(name, data_dict):
    print(f"\n==== {name.upper()} ====")
    for key, val in data_dict.items():
        print(f"- {key}: type={type(val)}, ", end="")
        if isinstance(val, torch.Tensor):
            print(f"shape={val.shape}, dtype={val.dtype}")
            print(f"  sample values: {val.flatten()[:10]}")
        elif isinstance(val, list):
            print(f"(LIST) len={len(val)} | type of first item: {type(val[0])}")
            if isinstance(val[0], torch.Tensor):
                print(f"  tensor[0] shape = {val[0].shape}")
                print(f"  values: {val[0].flatten()[:10]}")
        else:
            print(f"value: {val}")

def main():
    files = sorted(f for f in os.listdir(SUBGENRE_PATH) if f.endswith(".pt"))
    if len(files) < 3:
        print("Not enough files in subgenre for a triplet.")
        return

    # Assume files are in triplet order (as in your script)
    anchor = torch.load(os.path.join(SUBGENRE_PATH, files[0]), weights_only=True)
    positive = torch.load(os.path.join(SUBGENRE_PATH, files[1]), weights_only=True)
    negative = torch.load(os.path.join(SUBGENRE_PATH, files[2]), weights_only=True)

    unwrap_and_print("anchor", anchor)
    unwrap_and_print("positive", positive)
    unwrap_and_print("negative", negative)

if __name__ == "__main__":
    main()
