import runpod
import torch
from PIL import Image
import io
from transformers import AutoModelForObjectDetection, AutoProcessor
# minor change to trigger rebuild
model_id = "omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForObjectDetection.from_pretrained(model_id).eval().to("cuda")

DEFAULT_PROMPT = "a photo of people, cars, dogs, backpacks"

def handler(event):
    image_bytes = event["input"]["file_bytes"]
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = processor(images=image, text=DEFAULT_PROMPT, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections.append({
            "label": model.config.id2label[label.item()],
            "score": round(score.item(), 3),
            "box": [round(coord, 1) for coord in box.tolist()]
        })

    return detections

runpod.serverless.start({"handler": handler})
