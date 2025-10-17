#!/usr/bin/env python3

"""
Patch for PaddingCut num_frames calculation issue
"""

from lhotse.cut import PaddingCut
from lhotse.utils import compute_num_frames

# Store original PaddingCut.__init__
_original_padding_cut_init = PaddingCut.__init__

def patched_padding_cut_init(self, *args, **kwargs):
    """Patched PaddingCut.__init__ that ensures num_frames is properly calculated."""
    # Call original __init__
    _original_padding_cut_init(self, *args, **kwargs)
    
    # If num_frames is None, calculate it
    if self.num_frames is None and hasattr(self, 'duration') and hasattr(self, 'frame_shift'):
        if self.frame_shift is not None and self.frame_shift > 0:
            sampling_rate = getattr(self, 'sampling_rate', 16000)
            self.num_frames = compute_num_frames(
                duration=self.duration,
                frame_shift=self.frame_shift,
                sampling_rate=sampling_rate
            )
            print(f"🔧 Patched PaddingCut {self.id}: calculated num_frames={self.num_frames}")

def apply_padding_cut_patch():
    """Apply the patch to fix PaddingCut num_frames calculation."""
    PaddingCut.__init__ = patched_padding_cut_init
    print("🔧 Applied PaddingCut num_frames patch")

def test_patch():
    """Test the patch works correctly."""
    print("Testing PaddingCut patch...")
    
    # Test without patch first
    from lhotse.cut import PaddingCut as OriginalPaddingCut
    
    # Apply patch
    apply_padding_cut_patch()
    
    # Test with patch
    padding_cut = PaddingCut(
        id="test_patched_padding",
        duration=0.1,
        sampling_rate=16000,
        num_features=80,
        frame_shift=0.01,
        feat_value=-23.0
    )
    
    print(f"Patched PaddingCut:")
    print(f"  Duration: {padding_cut.duration}")
    print(f"  num_frames: {padding_cut.num_frames}")
    print(f"  Expected: {compute_num_frames(0.1, 0.01, 16000)}")

if __name__ == "__main__":
    test_patch()
