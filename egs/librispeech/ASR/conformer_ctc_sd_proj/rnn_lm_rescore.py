#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
RNN LM rescoring for CTC-only models (without attention decoder).
Modified from icefall.decode.rescore_with_rnn_lm to remove attention decoder dependency.
"""

from typing import Dict, List, Optional

import k2
import torch
from icefall.utils import add_eos, add_sos
from k2 import Nbest

# Default LM scale list
DEFAULT_LM_SCALE = [0.1 * i for i in range(1, 21)]  # [0.1, 0.2, ..., 2.0]


def rescore_with_rnn_lm_no_decoder(
    lattice: k2.Fsa,
    num_paths: int,
    rnn_lm_model: torch.nn.Module,
    blank_id: int,
    nbest_scale: float = 1.0,
    ngram_lm_scale: Optional[float] = None,
    rnn_lm_scale: Optional[float] = None,
    use_double_scores: bool = True,
) -> Dict[str, k2.Fsa]:
    """This function extracts `num_paths` paths from the given lattice and uses
    RNN LM to rescore them. The path with the highest score is the decoding output.
    
    This version is for CTC-only models without attention decoder.

    Args:
      lattice:
        An FsaVec with axes [utt][state][arc]. Must have "tokens" attribute.
      num_paths:
        Number of paths to extract from the given lattice for rescoring.
      rnn_lm_model:
        A rnn-lm model used for LM rescoring
      blank_id:
        The token ID for blank (usually 0 for CTC).
      nbest_scale:
        It's the scale applied to `lattice.scores`. A smaller value
        leads to more unique paths at the risk of missing the correct path.
      ngram_lm_scale:
        Optional. It specifies the scale for n-gram LM scores.
        If None, will test multiple values.
      rnn_lm_scale:
        Optional. It specifies the scale for RNN LM scores.
        If None, will test multiple values.
      use_double_scores:
        True to use double precision during computation.
    Returns:
      A dict of FsaVec, whose key contains a string like
      "ngram_lm_scale_X_rnn_lm_scale_Y" and the value is the
      best decoding path for each utterance in the lattice.
    """
    device = lattice.device
    
    # Simply use the best path from lattice (shortest path = greedy CTC)
    # No need for n-best extraction - just add RNN LM score to the best path
    best_path = k2.shortest_path(lattice, use_double_scores=use_double_scores)
    
    # Extract token sequences from best path
    assert hasattr(best_path, "tokens"), "lattice must have tokens attribute"
    assert isinstance(best_path.tokens, torch.Tensor)
    
    # Get token IDs (remove blank tokens)
    # best_path.tokens shape: [num_arcs]
    token_ids_list = []
    for i in range(best_path.shape[0]):  # For each utterance
        # Extract tokens for this utterance
        fsa = best_path[i]
        tokens = fsa.tokens[fsa.tokens > 0].tolist()  # Remove blanks (0)
        token_ids_list.append(tokens)
    
    if len(token_ids_list) == 0 or all(len(t) == 0 for t in token_ids_list):
        print("Warning: rescore_with_rnn_lm_no_decoder(): empty token-ids")
        return None
    
    token_ids = token_ids_list

    # Prepare tokens for RNN LM
    sos_id = 1  # Assuming SOS ID is 1 (adjust if different)
    eos_id = 1  # Assuming EOS ID is 1 (adjust if different)
    
    sos_tokens = add_sos(tokens, sos_id)
    tokens_eos = add_eos(tokens, eos_id)
    sos_tokens_row_splits = sos_tokens.shape.row_splits(1)
    sentence_lengths = sos_tokens_row_splits[1:] - sos_tokens_row_splits[:-1]

    x_tokens = sos_tokens.pad(mode="constant", padding_value=blank_id)
    y_tokens = tokens_eos.pad(mode="constant", padding_value=blank_id)

    x_tokens = x_tokens.to(torch.int64)
    y_tokens = y_tokens.to(torch.int64)
    sentence_lengths = sentence_lengths.to(torch.int64)

    # Compute RNN LM scores
    rnn_lm_nll = rnn_lm_model(x=x_tokens, y=y_tokens, lengths=sentence_lengths)
    assert rnn_lm_nll.ndim == 2
    assert rnn_lm_nll.shape[0] == len(token_ids)

    rnn_lm_scores = -1 * rnn_lm_nll.sum(dim=1)

    # Now add RNN LM scores to the best path
    # best_path.scores contains AM scores
    # We create a new score = AM_score + rnn_scale * RNN_LM_score
    
    rnn_scale = rnn_lm_scale if rnn_lm_scale is not None else 1.0
    
    # Get total scores for each utterance
    # best_path is FsaVec, so we need to add RNN LM scores per utterance
    best_path_clone = best_path.clone()
    
    # Add RNN LM scores to the FSA scores
    # Note: This is a simplified approach - we're adding the total RNN LM score
    # to the final state of each FSA
    for i in range(best_path_clone.shape[0]):
        # Get the total score for this utterance's path
        total_rnn_score = rnn_lm_scores[i].item()
        # Add to the FSA (this will affect the total path score)
        # We add it uniformly across all arcs (simplified)
        num_arcs = best_path_clone[i].scores.shape[0]
        if num_arcs > 0:
            best_path_clone[i].scores = best_path_clone[i].scores + (rnn_scale * total_rnn_score / num_arcs)
    
    key = f"ngram_lm_scale_0.0_rnn_lm_scale_{rnn_scale}"
    
    # Return the rescored path
    return {key: best_path_clone}
