import argparse
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO

class CarDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect_cars(self, image_path):
        results = self.model(image_path)
        detections = []
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, score, class_id = box.tolist()
                if int(class_id) == 2:  # Class ID for cars
                    detections.append((int(x1), int(y1), int(x2), int(y2)))
        return detections


class CarCounter:
    def __init__(self):
        self.lower_red1 = np.array([0, 120, 70])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 70])
        self.upper_red2 = np.array([180, 255, 255])

    def count_red_cars(self, image_path, detections):
        image = cv2.imread(image_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        red_count = 0
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        for (x1, y1, x2, y2) in detections:
            car_region = hsv[y1:y2, x1:x2]
            mask1 = cv2.inRange(car_region, self.lower_red1, self.upper_red1)
            mask2 = cv2.inRange(car_region, self.lower_red2, self.upper_red2)
            red_mask = mask1 + mask2

            if np.sum(red_mask) / (red_mask.shape[0] * red_mask.shape[1]) > 30:
                color = "r"
                red_count += 1
            else:
                color = "b"

            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        plt.title(f"Red cars count: {red_count}")
        plt.axis("off")
        plt.show()

        return len(detections), red_count


class ImageCaptioner:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_caption(self, image_path, car_count, red_count):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(
            **inputs, 
            max_length=80, 
            min_length=40, 
            num_beams=10, 
            repetition_penalty=1.2, 
            length_penalty=0.9,
            early_stopping=False
        )

        caption = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        caption += f" Number of cars found {car_count}, of which {red_count} are red."
        return caption


def main(image_path):
    # Step 1: Detect cars
    car_detector = CarDetector()
    detections = car_detector.detect_cars(image_path)

    # Step 2: Count red cars
    car_counter = CarCounter()
    car_count, red_count = car_counter.count_red_cars(image_path, detections)

    # Step 3: Generate caption
    image_captioner = ImageCaptioner()
    caption = image_captioner.generate_caption(image_path, car_count, red_count)
    
    print(f"Generated Caption: {caption}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Car detection, red car identification, and image captioning.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the image for processing.")
    
    args = parser.parse_args()
    main(args.image_path)
