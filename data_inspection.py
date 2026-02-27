import os

def print_counts(data_dir):
    for split in ['train', 'val', 'test']:
        print(f"\n--- {split.upper()} SET ---")
        path = os.path.join(data_dir, split)
        
        if not os.path.exists(path):
            print("Path not found:", path)
            continue
        
        for folder in os.listdir(path):
            class_path = os.path.join(path, folder)
            if os.path.isdir(class_path):
                count = len([
                    f for f in os.listdir(class_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                print(f"{folder}: {count} images")