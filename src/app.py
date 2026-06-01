from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO

import torch
import torch.nn as nn
from torchvision import transforms, models

from PIL import Image
import numpy as np
import json
import os
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Load class mapping
# =========================

with open("class_mapping.json", "r") as f:
    idx_to_class = json.load(f)

# =========================
# Load MobileNet
# =========================

mobilenet = models.mobilenet_v3_small(weights=None)

mobilenet.classifier[3] = nn.Linear(
    1024,
    26
)

mobilenet.load_state_dict(
    torch.load(
        "mobilenet_braille.pth",
        map_location="cpu"
    )
)

mobilenet.eval()

# =========================
# Load YOLO
# =========================

yolo = YOLO(r"C:\Users\Vansh Kapil\runs\obb\train-9\weights\best.pt")

# =========================
# Preprocessing
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# Routes
# =========================

@app.get("/")
def home():
    return {
        "message": "working"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    os.makedirs("debug", exist_ok=True)
    image = Image.open(
        file.file
    ).convert("RGB")

    image_np = np.array(image)

    # =========================
    # YOLO Detection
    # =========================

    results = yolo(
        image_np,
        conf=0.10
    )

    if results[0].obb is None:
        return {
            "text": "",
            "letters": [],
            "message": "No braille detected"
        }

    boxes = results[0].obb.xyxy

    if len(boxes) == 0:
        return {
            "text": "",
            "letters": [],
            "message": "No braille detected"
        }

    # =========================
    # Extract detections
    # =========================

    detections = []

    for box in boxes:

        x1, y1, x2, y2 = map(
            int,
            box.tolist()
        )

        detections.append(
            [x1, y1, x2, y2]
        )

    # =========================
    # Sort into rows
    # =========================

    detections.sort(
        key=lambda b: b[1]
    )

    avg_height = np.mean([
        b[3] - b[1]
        for b in detections
    ])

    row_threshold = avg_height * 0.8

    rows = []

    for box in detections:

        x1, y1, x2, y2 = box

        center_y = (y1 + y2) / 2

        added = False

        for row in rows:

            row_center = np.mean([
                (r[1] + r[3]) / 2
                for r in row
            ])

            if abs(center_y - row_center) < row_threshold:

                row.append(box)
                added = True
                break

        if not added:
            rows.append([box])

    # sort rows top -> bottom

    rows.sort(
        key=lambda row: np.mean([
            (b[1] + b[3]) / 2
            for b in row
        ])
    )

    # sort boxes left -> right

    for row in rows:
        row.sort(
            key=lambda b: b[0]
        )

    # =========================
    # Classification
    # =========================

    letters = []
    confidences = []

    text_rows = []

    for row in rows:

        row_text = ""

        gaps = []

        for i in range(len(row) - 1):

            gap = row[i + 1][0] - row[i][2]

            gaps.append(gap)

        avg_gap = np.mean(gaps) if gaps else 0

        for i, (x1, y1, x2, y2) in enumerate(row):

            crop = image.crop(
                (x1, y1, x2, y2)
            )
            crop.save(f"debug/{i}.png")

            x = transform(crop)
            x = x.unsqueeze(0)

            with torch.no_grad():

                output = mobilenet(x)

                probs = torch.softmax(
                    output,
                    dim=1
                )

                class_id = int(
                    torch.argmax(
                        probs,
                        dim=1
                    )
                )

                confidence = float(
                    torch.max(probs)
                )

            letter = idx_to_class[
                str(class_id)
            ]

            letters.append(letter)
            confidences.append(confidence)

            row_text += letter

            # =========================
            # Word spacing
            # =========================

            if i < len(row) - 1:

                next_x1 = row[i + 1][0]

                gap = next_x1 - x2

                if avg_gap > 0 and gap > avg_gap * 3.0:
                    row_text += " "

        text_rows.append(row_text)

    # =========================
    # Final text
    # =========================

    sentence = "\n".join(
        text_rows
    )

    return {
        "text": sentence,
        "letters": letters,
        "confidences": confidences,
        "num_characters": len(letters)
    }