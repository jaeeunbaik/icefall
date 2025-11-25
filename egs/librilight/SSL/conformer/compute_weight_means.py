import os
import torch
import pickle
import sys
from pathlib import PosixPath, WindowsPath

# 필요한 클래스들을 import하거나 정의
try:
    # EMATeacher 클래스를 찾아서 import
    sys.path.append('/home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/conformer_ctc_sd_proj')
    from ema_teacher import EMATeacher
except ImportError:
    # EMATeacher 클래스가 없다면 더미 클래스 정의
    class EMATeacher:
        def __init__(self, *args, **kwargs):
            pass

# 1. 모델 아키텍처 정의 (예시로 nn.Module 형태, 반드시 본인 모델로 바꿔야 함)
try:
    from conformer import Conformer  # 사용자 모델 정의 파일에서 import
except ImportError:
    pass

# 경로 타입과 필요한 커스텀 클래스를 safe globals에 추가
torch.serialization.add_safe_globals([PosixPath, WindowsPath])

# PrototypeKMeansManager를 allow-list에 등록 시도 (weights_only=True에서 필요할 수 있음)
try:
    # 현재 디렉터리 및 관련 레시피 경로를 우선적으로 탐색
    candidates = [
        os.path.dirname(os.path.abspath(__file__)),
        '/home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/conformer_ctc_sd_proj',
    ]
    for p in candidates:
        if p not in sys.path:
            sys.path.append(p)
    from k_means_clustering import PrototypeKMeansManager  # type: ignore
    torch.serialization.add_safe_globals([PrototypeKMeansManager])
except Exception:
    # 등록 실패해도 진행 (fallback 로직이 있음)
    pass

def safe_torch_load(path):
    """안전하게 체크포인트를 로드하는 함수"""
    try:
        # 먼저 weights_only=True로 시도 (PosixPath 허용)
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception as e1:
        print(f"weights_only=True로 로드 실패: {str(e1)[:200]}...")
        try:
            # weights_only=False로 시도
            try:
                return torch.load(path, map_location='cpu', weights_only=False)
            except ModuleNotFoundError as e_mod:
                # 필요한 모듈 경로를 추가 후 한 번 더 시도
                if 'k_means_clustering' in str(e_mod):
                    extra_paths = [
                        os.path.dirname(os.path.abspath(__file__)),
                        '/home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/conformer_ctc_sd_proj',
                    ]
                    for p in extra_paths:
                        if p not in sys.path:
                            sys.path.append(p)
                    try:
                        return torch.load(path, map_location='cpu', weights_only=False)
                    except Exception:
                        raise
                else:
                    raise
        except Exception as e2:
            print(f"일반 로드도 실패: {str(e2)[:200]}...")
            raise e2

# 2. 모델 파일 경로 리스트
def compute_mean_model(avg, epoch, model_fld, save_name):
    model_paths = []
    for i in range(avg):
        cur_epoch = epoch - i
        model_path = os.path.join(model_fld, f'epoch-{cur_epoch}.pt')
        model_paths.append(model_path)

    # 3. 첫 모델의 state_dict 구조 기준으로 초기화
    checkpoints = []
    for p in model_paths:
        print(f"로딩 중: {p}")
        try:
            ckpt = safe_torch_load(p)
            checkpoints.append(ckpt)
        except Exception as e:
            print(f"파일 {p} 로딩 실패: {e}")
            continue
    
    if not checkpoints:
        raise RuntimeError("로딩할 수 있는 체크포인트가 없습니다.")
    
    print(f"{len(checkpoints)}개 체크포인트 로딩 완료")
    
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

    # 5. 최소 안전 포맷으로 저장: 모델 state_dict만 저장
    #    (추가적인 커스텀 객체나 경로 객체를 포함하지 않도록 함)
    safe_checkpoint = {
        'model': avg_state_dict,
        # 선택적으로 메타데이터를 넣고 싶다면 텐서/프리미티브 타입만 사용하세요.
        # 'meta': {'averaged_from': model_paths, 'avg': len(state_dicts)}
    }

    # 6. 평균 모델 저장 (model state_dict만 포함)
    torch.save(safe_checkpoint, save_name)
    print(f"평균 모델 저장 완료(안전 포맷): {save_name}")
    
if __name__=='__main__':
    avg = 5
    epoch = 11
    model_fld = '/home/hdd2/jenny/ASRToolkit/icefall/egs/librilight/SSL/conformer/librilight/pretrain_random_init_recluster-2_tw80_6,12,18'
    save_name = 'pretrain_random_init_recluster-2_tw80_6,12,18_avg5.pt'  # avg11에서 avg3으로 변경
    compute_mean_model(avg, epoch, model_fld, save_name)