import os
import shutil
import glob

DATA_ROOT = "data/raw_datasets"

def move_contents(src_dir, dest_dir):
    if not os.path.exists(src_dir):
        print(f"Source {src_dir} does not exist, skipping.")
        return
    
    print(f"Moving contents from {src_dir} to {dest_dir}...")
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dest_dir, item)
        if os.path.exists(d):
            print(f"  Warning: {d} already exists, skipping {item}")
        else:
            shutil.move(s, d)
    
    # Try to remove the now empty source directories (and their parents if empty)
    try:
        os.rmdir(src_dir)
        parent = os.path.dirname(src_dir)
        if not os.listdir(parent):
            os.rmdir(parent)
    except:
        pass

def clean_metadata(root_dir):
    print(f"Cleaning metadata in {root_dir}...")
    patterns = [".mdl", ".msc", ".mv", ".gitattributes", ".lock", "._____temp"]
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if any(f.endswith(p) or f == p for p in patterns):
                os.remove(os.path.join(root, f))
        for d in dirs:
             if any(d.endswith(p) or d == p for p in patterns):
                shutil.rmtree(os.path.join(root, d))

# 1. MetaMathQA
# Structure: metamathqa/swift/MetaMathQA/* -> metamathqa/*
move_contents(
    os.path.join(DATA_ROOT, "metamathqa/swift/MetaMathQA"),
    os.path.join(DATA_ROOT, "metamathqa")
)

# 2. Math (Competition Math)
# Structure: math/modelscope/competition_math/* -> math/*
move_contents(
    os.path.join(DATA_ROOT, "math/modelscope/competition_math"),
    os.path.join(DATA_ROOT, "math")
)

# 3. Commonsense170k
# Structure: commonsense170k/deepmath/commonsense_170k/* -> commonsense170k/*
move_contents(
    os.path.join(DATA_ROOT, "commonsense170k/deepmath/commonsense_170k"),
    os.path.join(DATA_ROOT, "commonsense170k")
)

# 4. Clean up metadata files everywhere
clean_metadata(DATA_ROOT)

print("Organization complete.")
