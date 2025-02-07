from PIL import Image
import os

path = "./data/image_generation"

for item in os.listdir(path):
    item_path = os.path.join(path, item)

    if os.path.isdir(item_path):
        input_folder = item_path

        for filename in os.listdir(input_folder):
            if filename.lower().endswith(".webp"):
                webp_path = os.path.join(input_folder, filename)
                png_path = os.path.join(input_folder, filename.replace(".webp", ".png"))

                try:
                    with Image.open(webp_path) as img:
                        img.save(png_path, "PNG")
                        print(f"Converted: {filename} → {png_path}")

                    os.remove(webp_path)
                    print(f"Deleted: {filename}")

                except Exception as e:
                    print(f"Error converting {filename}: {e}")

print("Conversion completed!")
