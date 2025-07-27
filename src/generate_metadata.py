import os
import pandas as pd

def get_image_paths(root_dir, category):
    data = []
    root_dir = os.path.abspath(root_dir)

    for subset in ['train', 'test']:
        subset_path = os.path.join(root_dir, category, subset)
        if not os.path.exists(subset_path):
            continue

        for defect_type in os.listdir(subset_path):
            defect_path = os.path.join(subset_path, defect_type)
            if not os.path.isdir(defect_path):
                continue

            for img_file in os.listdir(defect_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    label = 'non-defective' if defect_type == 'good' else 'defective'
                    rel_path = os.path.join(category, subset, defect_type, img_file)
                    data.append({
                        'path': rel_path.replace("\\", "/"),  # for consistency
                        'label': label,
                        'defect_type': defect_type
                    })

    return data

if __name__ == "__main__":
    dataset_root = 'data'  # bottle and tiles data under data/bottle and data/tile under this
    categories = ['bottle', 'tile']

    all_data = []

    for cat in categories:
        all_data.extend(get_image_paths(dataset_root, cat))

    df = pd.DataFrame(all_data)
    df.to_csv('metadata.csv', index=False)
    print(" metadata.csv created successfully with", len(df), "entries.")
