import os

ROOT_DIR = r"path_to_processed"  

def contar_imagens_em_subpastas(root_dir):
    for root, dirs, files in os.walk(root_dir):
        img_count = sum(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files)
        if img_count > 0:
            print(f'{root} -> {img_count} imagens')

contar_imagens_em_subpastas(ROOT_DIR)
