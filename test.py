import torch
from ultralytics import YOLO

# 1. 로컬에 있는 'best.pt'를 불러옵니다.
model = YOLO('best.pt')

# 2. 서버가 좋아하는 '투명 봉투(가중치 데이터)'로 새로 저장합니다.
# 이 옵션이 서버 배포의 핵심입니다!
torch.save(model.model.state_dict(), 'best_fix.pt', _use_new_zipfile_serialization=False)

print("✅ 'best_fix.pt' 생성 성공! 이제 이 파일을 깃허브에 올리세요.")