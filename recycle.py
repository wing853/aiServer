from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import cv2
import tempfile
import jwt  # PyJWT 필요
from functools import wraps
import torch  # 추가: PyTorch 보안 설정용
import ultralytics # 추가

app = Flask(__name__)

# --- PyTorch 2.6+ 보안 로드 오류 해결 코드 ---
# YOLO 모델 클래스를 허용 리스트에 추가합니다.
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([
        ultralytics.nn.tasks.DetectionModel,
        torch.nn.modules.container.Sequential
    ])
# ------------------------------------------

# 시크릿 키 (JWT 검증용)
JWT_SECRET = os.environ.get("JWT_SECRET", "my-super-secure-and-long-secret-key-1234567890")

# 모델 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT = os.path.join(BASE_DIR, "best.pt")

# 안전하게 모델 로드 시도
try:
    # 최신 버전에서는 weights_only=False를 명시하거나 위에서 safe_globals를 설정해야 함
    model = YOLO(WEIGHT)
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    # 최후의 수단: 환경 변수로 보안 체크를 끌 수도 있지만, 위 설정이 더 권장됩니다.
    # os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
    model = YOLO(WEIGHT)

DISPOSAL = {
    "캔":      "캔 전용 수거함에 버려주세요.",
    "플라스틱":  "플라스틱 전용 수거함에 버려주세요.",
    "종이":    "종이류 전용 수거함에 버려주세요.",
    "비닐":    "비닐 전용 수거함에 버려주세요",
    "페트병":  "페트병 전용 수거함에 버려주세요",
    "유리병":  "유리 전용 수거함에 버려주세요",
    "스티로폼": "스티로폼 전용 수거함에 버려주세요"
}

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "권한 없음: Authorization 헤더 필요"}), 403
        token = auth_header.split(" ")[1]
        try:
            jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "토큰 만료"}), 403
        except jwt.InvalidTokenError:
            return jsonify({"error": "유효하지 않은 토큰"}), 403
        return f(*args, **kwargs)
    return decorated

@app.route("/", methods=["GET"])
def test():
    return "AI 분석 서버 접속 성공 (Live)"

@app.route("/recycle/analyze", methods=["POST"])
# @token_required  # 테스트 완료 후 주석 해제하세요
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "이미지 파일이 필요합니다."}), 400

    image_file = request.files['image']
    if image_file.filename == "":
        return jsonify({"error": "파일 이름이 비어 있습니다."}), 400

    # 임시 파일 저장 및 처리
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            image_file.save(temp.name)
            temp_path = temp.name
        
        img = cv2.imread(temp_path)
        if img is None:
             return jsonify({"error": "이미지를 읽을 수 없습니다."}), 400

        # 모델 추론
        results = model(img, verbose=False)
        
        if not results or len(results[0].boxes) == 0:
            return jsonify({"error": "객체 감지 실패 (분석할 수 없는 이미지)"}), 400

        # 신뢰도가 가장 높은 결과 추출
        best = max(results[0].boxes, key=lambda b: float(b.conf[0]))
        cls_id = int(best.cls[0])
        confidence = float(best.conf[0])
        category = model.names[cls_id]
        disposal = DISPOSAL.get(category, "일반 쓰레기통에 버려주세요.")

        return jsonify({
            "category": category,
            "confidence": round(confidence, 4),
            "disposal_method": disposal
        })

    except Exception as e:
        print(f"분석 중 서버 오류: {e}")
        return jsonify({"error": "서버 내부 오류 발생"}), 500
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    # Render는 환경 변수 PORT를 사용하므로 아래 설정이 중요합니다.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)