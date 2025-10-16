import os
import torch

# 1. 모델 아키텍처 정의 (예시로 nn.Module 형태, 반드시 본인 모델로 바꿔야 함)
from conformer import Conformer  # 사용자 모델 정의 파일에서 import

# 2. 모델 파일 경로 리스트
def compute_mean_model(avg, epoch, model_fld):
    model_paths = []
    for i in range(avg):
        cur_epoch = epoch - i
        model_path = os.path.join(model_fld, f'epoch-{cur_epoch}.pt')
        model_paths.append(model_path)

    # 3. 첫 모델의 state_dict 구조 기준으로 초기화
    checkpoints = [torch.load(p, map_location='cpu', weights_only=False) for p in model_paths]
    
    # Extract only the model state dict from each checkpoint
    state_dicts = []
    for ckpt in checkpoints:
        if 'model' in ckpt:
            state_dicts.append(ckpt['model'])
        else:
            # If the checkpoint is already a state dict
            state_dicts.append(ckpt)
    
    avg_state_dict = {}

    # 4. 파라미터별 평균 계산
    for key in state_dicts[0].keys():
        # 모든 모델에서 해당 key의 파라미터를 스택해서 평균
        stacked = torch.stack([sd[key] for sd in state_dicts], dim=0)
        
        # Check if the parameter is floating point (can be averaged)
        if stacked.dtype.is_floating_point or stacked.dtype.is_complex:
            avg_param = torch.mean(stacked, dim=0)
        else:
            # For integer types (like embedding indices), use the first model's value
            print(f"Skipping averaging for {key} (dtype: {stacked.dtype}) - using first model's value")
            avg_param = state_dicts[0][key]
        
        avg_state_dict[key] = avg_param

    # 5. 기존 체크포인트 구조를 복제하고 model 부분만 교체
    # 첫 번째 체크포인트의 전체 구조를 기준으로 복제
    base_checkpoint = checkpoints[0].copy()
    
    # model 부분만 평균된 state_dict로 교체
    base_checkpoint['model'] = avg_state_dict
    
    # 6. 평균 모델 저장 (전체 체크포인트 구조 유지)
    torch.save(base_checkpoint, 'conformer_ctc_70000_from77avg10.pt')
    print("평균 모델 저장 완료: conformer_ctc_70000_from77avg10.pt")
    
if __name__=='__main__':
    avg = 10
    epoch = 77
    model_fld = '/home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/conformer_ctc/exp_70000'
    compute_mean_model(avg, epoch, model_fld)