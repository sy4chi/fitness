import onnx
import os

models_dir = "/Users/arthur/PyCharmMiscProject/models"
exercises = ["squat", "pushup", "pullup"]

for exercise in exercises:
    onnx_path = os.path.join(models_dir, f"{exercise}.onnx")
    print(f"Checking {onnx_path}...")
    if not os.path.exists(onnx_path):
        print(f"File not found: {onnx_path}")
        continue
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"{exercise}.onnx is VALID!")
    except Exception as e:
        print(f"{exercise}.onnx is INVALID: {e}")
