import os
import torch
import warnings

# [1단계] 모든 보안 경고 및 검사 강제 비활성화
# PyTorch 2.6+의 Pickle 로드 보안 정책을 우회하기 위한 최우선 설정입니다.
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
warnings.filterwarnings("ignore")

try:
    import ultralytics
    import collections
    
    # 보안 정책 대응을 위한 안전 리스트 등록
    if hasattr(torch.serialization, 'add_safe_globals'):
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.modules.block import C2f, Bottleneck, SPPF
        from ultralytics.nn.modules.head import Detect
        
        torch.serialization.add_safe_globals([
            DetectionModel, Conv, C2f, Bottleneck, SPPF, Detect,
            torch.nn.modules.pooling.MaxPool2d,
            torch.nn.modules.container.Sequential,
            torch.nn.modules.container.ModuleList,
            torch.nn.modules.conv.Conv2d,
            torch.nn.modules.batchnorm.BatchNorm2d,
            torch.nn.modules.activation.SiLU,
            torch.nn.modules.activation.LeakyReLU,
            torch.nn.modules.upsampling.Upsample,
            torch.storage._load_from_bytes,
            torch._utils._rebuild_tensor_v2,
            torch.Size,
            collections.OrderedDict
        ])
except Exception as e:
    print(f"⚠️ 초기 보안 설정 중 참고: {e}")

from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import tempfile

app = Flask(__name__)

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT = os.path.join(BASE_DIR, "best.pt")

# [2단계] 모델 로드 실행 (들여쓰기 오류 수정 완료)
model = None
try:
    # A. 표준 방식으로 우선 시도
    model = YOLO(WEIGHT)
    print("✅✅ 모델 로드 성공!")
except Exception as e:
    print(f"⚠️ 표준 로드 실패, 보안 강제 해제 시도 중: {e}")
    try:
        # B. 최후의 수단: weights_only=False로 강제 로드
        # 중첩된 try-except문의 들여쓰기를 정확히 맞췄습니다.
        ckpt = torch.load(WEIGHT, map_location='cpu', weights_only=False)
        model = YOLO(WEIGHT)
        
        # 모델 객체에 가중치 수동 주입
        if isinstance(ckpt, dict) and 'model' in ckpt:
            state_dict = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
            model.model.load_state_dict(state_dict)
        
        print("✅✅ [최종] 보안 해제 강제 로드 성공!")
    except Exception as final_e:
        print(f"❌ 모델 로드 최종 실패: {final_e}")

# 분리 배출 데이터 가이드
DISPOSAL = {
    "캔": "캔 전용 수거함에 버려주세요.",
    "플라스틱": "플라스틱 전용 수거함에 버려주세요.",
    "종이": "종이류 전용 수거함에 버려주세요.",
    "비닐": "비닐 전용 수거함에 버려주세요.",
    "페트병": "페트병 전용 수거함에 버려주세요.",
    "유리병": "유리 전용 수거함에 버려주세요.",
    "스티로폼": "스티로폼 전용 수거함에 버려주세요."
}

@app.route("/", methods=["GET"])
def health():
    if model:
        return "✅ AI Server is Live and Model is Loaded!"
    else:
        return "⚠️ AI Server is Live but Model Load Failed."

@app.route("/recycle/analyze", methods=["POST"])
def analyze_image():
    if model is None:
        return jsonify({"error": "모델이 로드되지 않았습니다."}), 500
    
    if 'image' not in request.files:
        return jsonify({"error": "이미지 파일이 없습니다."}), 400

    image_file = request.files['image']
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            image_file.save(temp.name)
            temp_path = temp.name
        
        img = cv2.imread(temp_path)
        if img is None:
            return jsonify({"error": "이미지 읽기 실패"}), 400

        results = model(img, verbose=False)
        
        if not results or len(results[0].boxes) == 0:
            return jsonify({"error": "감지된 쓰레기 객체가 없습니다."}), 400

        best = max(results[0].boxes, key=lambda b: float(b.conf[0]))
        category = model.names[int(best.cls[0])]
        
        return jsonify({
            "category": category,
            "confidence": round(float(best.conf[0]), 4),
            "disposal_method": DISPOSAL.get(category, "일반 쓰레기로 배출하세요.")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    # Render 환경의 포트 바인딩 (기본 10000)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)