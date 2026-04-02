from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import cv2
import tempfile
import jwt  # PyJWT 필요
from functools import wraps

app = Flask(__name__)

# 시크릿 키 (JWT 검증용)
JWT_SECRET = os.environ.get("JWT_SECRET", "my-super-secure-and-long-secret-key-1234567890")  # Render 환경변수에 넣으면 안전

# 모델 로드 (서버 실행 시 1회)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT = os.path.join(BASE_DIR, "best.pt")
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

# JWT 인증 데코레이터
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
@app.route("/", methods = ["GET"])
def test():
    return "서버 접속됨"

@app.route("/recycle/analyze", methods=["POST"])
#@token_required
def analyze_image():
    # 요청 헤더 확인
    print("==== Request Headers ====")
    for k, v in request.headers.items():
        print(f"{k}: {v}")
    
    # 요청 파일 확인
    print("==== Request Files ====")
    for k, v in request.files.items():
        print(f"{k}: {v.filename}")
    
    if 'image' not in request.files:
        return jsonify({"error": "이미지 파일이 필요합니다."}), 400

    image_file = request.files['image']
    if image_file.filename == "":
        return jsonify({"error": "파일 이름이 비어 있습니다."}), 400

    # 임시 파일 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image_file.save(temp.name)
        img = cv2.imread(temp.name)

    if img is None:
        return jsonify({"error": "이미지를 읽을 수 없습니다."}), 400

    # 모델 추론
    results = model(img, verbose=False)
    if len(results[0].boxes) == 0:
        return jsonify({"error": "객체 감지 실패"}), 400

    best = max(results[0].boxes, key=lambda b: float(b.conf[0]))
    cls_id = int(best.cls[0])
    confidence = float(best.conf[0])
    category = model.names[cls_id]
    disposal = DISPOSAL.get(category, "일반 쓰레기통에 버려주세요.")

    os.remove(temp.name)  # 임시 파일 삭제

    print(f"Analyzed category: {category}, confidence: {confidence}")

    return jsonify({
        "category": category,
        "confidence": round(confidence, 4),
        "disposal_method": disposal
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)