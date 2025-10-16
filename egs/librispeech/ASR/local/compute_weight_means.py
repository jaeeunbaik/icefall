import os
import torch

# 1. 모델 아키텍처 정의 (예시로 nn.Module 형태, 반드시 본인 모델로 바꿔야 함)
from .conformer_ctc.conformer import Conformer  # 사용자 모델 정의 파일에서 import

# 2. 모델 파일 경로 리스트
def compute_mean_model(avg, epoch, model_fld):
    model_paths = []
    for i in range(avg):
        cur_epoch = epoch - i
        model_path = os.path.join(model_fld, f'epoch-{cur_epoch}.pt')
        model_paths.append(model_path)

    # 3. 첫 모델의 state_dict 구조 기준으로 초기화
    state_dicts = [torch.load(p, map_location='cpu') for p in model_paths]
    avg_state_dict = {}

    # 4. 파라미터별 평균 계산
    for key in state_dicts[0].keys():
        # 모든 모델에서 해당 key의 파라미터를 스택해서 평균
        stacked = torch.stack([sd[key] for sd in state_dicts], dim=0)
        avg_param = torch.mean(stacked, dim=0)
        avg_state_dict[key] = avg_param

    # 5. 새 모델 생성 및 평균 파라미터 적용
    model = Conformer()
    model.load_state_dict(avg_state_dict)

    # 6. 평균 모델 저장
    torch.save(model.state_dict(), 'average_model.pt')
    print("평균 모델 저장 완료: average_model.pt")
    
if __name__=='__main__':
    avg = 5
    epoch = 9
    model_fld = '/home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/conformer_ctc/exp_70000'
    compute_mean_model(avg, epoch, model_fld)