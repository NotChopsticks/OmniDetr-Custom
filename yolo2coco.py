import os
import json

def yolo_to_coco(yolo_dir, img_dir, output_file, categories):
    data = {}
    data["images"] = []
    data["type"] = "instances"
    data["annotations"] = []
    data["categories"] = [{"id": i, "name": name} for i, name in enumerate(categories)]
    
    image_id = 0
    annotation_id = 0

    for filename in os.listdir(yolo_dir):
        if filename.endswith(".txt"):
            image_name = filename.replace(".txt", ".jpeg")
            
            # Assuming all images have the same size
            # This can be modified to read actual image dimensions if needed
            width, height = 1280, 720  # Replace with your image dimensions

            data["images"].append({
                "file_name": image_name,
                "height": height,
                "width": width,
                "id": image_id
            })

            with open(os.path.join(yolo_dir, filename), 'r') as file:
                for line in file:
                    line = line.strip()
                    parts = line.split()
                    category_id, x_center, y_center, box_width, box_height = map(float, parts)
                    x_top_left = (x_center - box_width / 2) * width
                    y_top_left = (y_center - box_height / 2) * height
                    box_width *= width
                    box_height *= height

                    data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(category_id),
                        "bbox": [x_top_left, y_top_left, box_width, box_height],
                        "area": box_width * box_height,
                        "iscrowd": 0,
                        "segmentation": []
                    })

                    annotation_id += 1

            image_id += 1

    with open(output_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)

# Example usage
yolo_dir = '~/UNCC/10_aug_sp_cart_red_person_data/labels/train'
img_dir = '~/UNCC/10_aug_sp_cart_red_person_data/images/train'
output_file = '10_aug_sp_cart_red_person_data_COCO_train.json'
categories = ["STORE_PRODUCT", "RED_VEST", "PERSON", "CART", "HANDSCANNER"]

yolo_to_coco(yolo_dir, img_dir, output_file, categories)
