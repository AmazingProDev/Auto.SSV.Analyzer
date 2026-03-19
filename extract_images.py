import zipfile
import os
import sys

def extract_xlsx_images(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    out_dir = file_path.rsplit('.', 1)[0] + "_extracted"
    os.makedirs(out_dir, exist_ok=True)
    
    extracted_count = 0
    with zipfile.ZipFile(file_path, 'r') as z:
        media_files = [n for n in z.namelist() if n.startswith('xl/media/')]
        for f in media_files:
            filename = os.path.basename(f)
            if filename:  # Skip folder definitions
                with z.open(f) as source, open(os.path.join(out_dir, filename), "wb") as target:
                    target.write(source.read())
                extracted_count += 1
                    
    print(f"Extracted {extracted_count} images to '{out_dir}/'")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        extract_xlsx_images(sys.argv[1])
    else:
        print("Usage: python3 extract_images.py <file.xlsx>")
