import os
import torch
import warnings
import collections

# [1단계] 최우선 보안 및 환경 설정
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
warnings.filterwarnings("ignore")

try:
    if hasattr(torch.serialization, 'add_safe_globals'):
        # YOLO 내부에서 사용하는 모든 핵심 클래스를 명시적으로 허용
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.nn.modules import Conv, C2f, Bottleneck, SPPF, Detect
        
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
    print(f"⚠️ 초기 설정 참고: {e}")

from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import tempfile

app = Flask(__name__)

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_FIX = os.path.join(BASE_DIR, "best_fix.pt")
WEIGHT_ORIGIN = os.path.join(BASE_DIR, "best.pt")

model = None

def load_model_strategy():
    global model
    try:
        if os.path.exists(WEIGHT_FIX):
            print(f"🔄 {WEIGHT_FIX}를 발견했습니다. 가중치 주입 방식을 시작합니다.")
            # 1. 기본 모델 구조를 먼저 생성 (yolov8n.pt는 필요시 자동 다운로드됨)
            model = YOLO('yolov8n.pt') 
            
            # 2. 가중치 데이터만 강제로 로드
            ckpt = torch.load(WEIGHT_FIX, map_location='cpu', weights_only=False)
            
            # 3. 가중치 추출 (로컬 저장 방식에 따른 다양한 케이스 대응)
            if isinstance(ckpt, dict):
                if 'model' in ckpt:
                    # model 객체가 들어있는 경우
                    state_dict = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
                elif 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                else:
                    state_dict = ckpt
            else:
                state_dict = ckpt
            
            # 4. 모델에 가중치 덮어씌우기
            model.model.load_state_dict(state_dict, strict=False)
            print("✅✅ [성공] 가중치 수동 주입 완료!")
        
        elif os.path.exists(WEIGHT_ORIGIN):
            print(f"⚠️ {WEIGHT_FIX}가 없습니다. 기존 방식으로 {WEIGHT_ORIGIN} 로드를 시도합니다.")
            model = YOLO(WEIGHT_ORIGIN)
            print("✅✅ [성공] 원본 모델 로드 완료!")
        else:
            print("❌ 로드할 모델 파일(.pt)이 하나도 없습니다!")

    except Exception as e:
        print(f"❌ 모델 로드 최종 실패: {e}")
        model = None

# 서버 기동 시 로드 실행
load_model_strategy()

DISPOSAL = {
    "캔": "캔 전용 수거함에 버려주세요.",
    "플라스틱": "플라스틱 전용 수거함에 버려주세요.",
    "종일": "종이류 전용 수거함에 버려주세요.", # 오타 수정: 종일 -> 종이
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
        load_model_strategy()
        return "⚠️ AI Server is Live but Model Load Failed. Retrying..."

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
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)