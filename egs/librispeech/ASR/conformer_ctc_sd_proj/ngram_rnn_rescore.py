#!/usr/bin/env python3
"""
Combined N-gram + RNN LM rescoring for BPE-CTC models.
This module provides functions to:
1. Extract n-best paths from CTC lattice
2. Rescore with n-gram LM
3. Further rescore with RNN LM
4. Select the best path based on combined scores
"""

import logging
from typing import Dict, List, Optional, Tuple

import k2
import torch
from icefall.utils import add_eos, add_sos, get_texts


def _intersect_device(
    a_fsas: k2.Fsa,
    b_fsas: k2.Fsa,
    b_to_a_map: torch.Tensor,
    sorted_match_a: bool,
    batch_size: int = 50,
) -> k2.Fsa:
    """This is a wrapper of k2.intersect_device and its purpose is to split
    b_fsas into several batches and process each batch separately to avoid
    CUDA OOM error.

    The arguments and return value of this function are the same as
    :func:`k2.intersect_device`.
    """
    num_fsas = b_fsas.shape[0]
    if num_fsas <= batch_size:
        return k2.intersect_device(
            a_fsas, b_fsas, b_to_a_map=b_to_a_map, sorted_match_a=sorted_match_a
        )

    num_batches = (num_fsas + batch_size - 1) // batch_size
    splits = []
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, num_fsas)
        splits.append((start, end))

    ans = []
    for start, end in splits:
        indexes = torch.arange(start, end).to(b_to_a_map)

        fsas = k2.index_fsa(b_fsas, indexes)
        b_to_a = k2.index_select(b_to_a_map, indexes)
        path_lattice = k2.intersect_device(
            a_fsas, fsas, b_to_a_map=b_to_a, sorted_match_a=sorted_match_a
        )
        ans.append(path_lattice)

    return k2.cat(ans)


def get_lattice(
    nnet_output: torch.Tensor,
    decoding_graph: k2.Fsa,
    supervision_segments: torch.Tensor,
    search_beam: float,
    output_beam: float,
    min_active_states: int,
    max_active_states: int,
    subsampling_factor: int = 1,
) -> k2.Fsa:
    """Get the decoding lattice from a decoding graph and neural
    network output.
    Args:
      nnet_output:
        It is the output of a neural model of shape `(N, T, C)`.
      decoding_graph:
        An Fsa, the decoding graph. It can be either an HLG
        (see `compile_HLG.py`) or an H (see `k2.ctc_topo`).
      supervision_segments:
        A 2-D **CPU** tensor of dtype `torch.int32` with 3 columns.
        Each row contains information for a supervision segment. Column 0
        is the `sequence_index` indicating which sequence this segment
        comes from; column 1 specifies the `start_frame` of this segment
        within the sequence; column 2 contains the `duration` of this
        segment.
      search_beam:
        Decoding beam, e.g. 20.  Smaller is faster, larger is more exact
        (less pruning). This is the default value; it may be modified by
        `min_active_states` and `max_active_states`.
      output_beam:
         Beam to prune output, similar to lattice-beam in Kaldi.  Relative
         to best path of output.
      min_active_states:
        Minimum number of FSA states that are allowed to be active on any given
        frame for any given intersection/composition task. This is advisory,
        in that it will try not to have fewer than this number active.
        Set it to zero if there is no constraint.
      max_active_states:
        Maximum number of FSA states that are allowed to be active on any given
        frame for any given intersection/composition task. This is advisory,
        in that it will try not to exceed that but may not always succeed.
        You can use a very large number if no constraint is needed.
      subsampling_factor:
        The subsampling factor of the model.
    Returns:
      An FsaVec containing the decoding result. It has axes [utt][state][arc].
    """
    dense_fsa_vec = k2.DenseFsaVec(
        nnet_output,
        supervision_segments,
        allow_truncate=subsampling_factor - 1,
    )

    lattice = k2.intersect_dense_pruned(
        decoding_graph,
        dense_fsa_vec,
        search_beam=search_beam,
        output_beam=output_beam,
        min_active_states=min_active_states,
        max_active_states=max_active_states,
    )

    return lattice

class Nbest(object):
    """
    An Nbest object contains two fields:

        (1) fsa. It is an FsaVec containing a vector of **linear** FSAs.
                 Its axes are [path][state][arc]
        (2) shape. Its type is :class:`k2.RaggedShape`.
                   Its axes are [utt][path]

    The field `shape` has two axes [utt][path]. `shape.dim0` contains
    the number of utterances, which is also the number of rows in the
    supervision_segments. `shape.tot_size(1)` contains the number
    of paths, which is also the number of FSAs in `fsa`.

    Caution:
      Don't be confused by the name `Nbest`. The best in the name `Nbest`
      has nothing to do with `best scores`. The important part is
      `N` in `Nbest`, not `best`.
    """

    def __init__(self, fsa: k2.Fsa, shape: k2.RaggedShape) -> None:
        """
        Args:
          fsa:
            An FsaVec with axes [path][state][arc]. It is expected to contain
            a list of **linear** FSAs.
          shape:
            A ragged shape with two axes [utt][path].
        """
        assert len(fsa.shape) == 3, f"fsa.shape: {fsa.shape}"
        assert shape.num_axes == 2, f"num_axes: {shape.num_axes}"

        if fsa.shape[0] != shape.tot_size(1):
            raise ValueError(
                f"{fsa.shape[0]} vs {shape.tot_size(1)}\n"
                "Number of FSAs in `fsa` does not match the given shape"
            )

        self.fsa = fsa
        self.shape = shape

    def __str__(self):
        s = "Nbest("
        s += f"Number of utterances:{self.shape.dim0}, "
        s += f"Number of Paths:{self.fsa.shape[0]})"
        return s

    @staticmethod
    def from_lattice(
        lattice: k2.Fsa,
        num_paths: int,
        use_double_scores: bool = True,
        nbest_scale: float = 0.5,
    ) -> "Nbest":
        """Construct an Nbest object by **sampling** `num_paths` from a lattice.

        Each sampled path is a linear FSA.

        We assume `lattice.labels` contains token IDs and `lattice.aux_labels`
        contains word IDs.

        Args:
          lattice:
            An FsaVec with axes [utt][state][arc].
          num_paths:
            Number of paths to **sample** from the lattice
            using :func:`k2.random_paths`.
          use_double_scores:
            True to use double precision in :func:`k2.random_paths`.
            False to use single precision.
          scale:
            Scale `lattice.score` before passing it to :func:`k2.random_paths`.
            A smaller value leads to more unique paths at the risk of being not
            to sample the path with the best score.
        Returns:
          Return an Nbest instance.
        """
        saved_scores = lattice.scores.clone()
        lattice.scores *= nbest_scale
        # path is a ragged tensor with dtype torch.int32.
        # It has three axes [utt][path][arc_pos]
        path = k2.random_paths(
            lattice, num_paths=num_paths, use_double_scores=use_double_scores
        )
        lattice.scores = saved_scores

        # word_seq is a k2.RaggedTensor sharing the same shape as `path`
        # but it contains word IDs. Note that it also contains 0s and -1s.
        # The last entry in each sublist is -1.
        # It axes is [utt][path][word_id]
        if isinstance(lattice.aux_labels, torch.Tensor):
            word_seq = k2.ragged.index(lattice.aux_labels, path)
        else:
            word_seq = lattice.aux_labels.index(path)
            word_seq = word_seq.remove_axis(word_seq.num_axes - 2)
        word_seq = word_seq.remove_values_leq(0)

        # Each utterance has `num_paths` paths but some of them transduces
        # to the same word sequence, so we need to remove repeated word
        # sequences within an utterance. After removing repeats, each utterance
        # contains different number of paths
        #
        # `new2old` is a 1-D torch.Tensor mapping from the output path index
        # to the input path index.
        _, _, new2old = word_seq.unique(
            need_num_repeats=False, need_new2old_indexes=True
        )

        # kept_path is a ragged tensor with dtype torch.int32.
        # It has axes [utt][path][arc_pos]
        kept_path, _ = path.index(new2old, axis=1, need_value_indexes=False)

        # utt_to_path_shape has axes [utt][path]
        utt_to_path_shape = kept_path.shape.get_layer(0)

        # Remove the utterance axis.
        # Now kept_path has only two axes [path][arc_pos]
        kept_path = kept_path.remove_axis(0)

        # labels is a ragged tensor with 2 axes [path][token_id]
        # Note that it contains -1s.
        labels = k2.ragged.index(lattice.labels.contiguous(), kept_path)

        # Remove -1 from labels as we will use it to construct a linear FSA
        labels = labels.remove_values_eq(-1)

        if isinstance(lattice.aux_labels, k2.RaggedTensor):
            # lattice.aux_labels is a ragged tensor with dtype torch.int32.
            # It has 2 axes [arc][word], so aux_labels is also a ragged tensor
            # with 2 axes [arc][word]
            aux_labels, _ = lattice.aux_labels.index(
                indexes=kept_path.values, axis=0, need_value_indexes=False
            )
        else:
            assert isinstance(lattice.aux_labels, torch.Tensor)
            aux_labels = k2.index_select(lattice.aux_labels, kept_path.values)
            # aux_labels is a 1-D torch.Tensor. It also contains -1 and 0.

        fsa = k2.linear_fsa(labels)
        fsa.aux_labels = aux_labels
        # Caution: fsa.scores are all 0s.
        # `fsa` has only one extra attribute: aux_labels.
        return Nbest(fsa=fsa, shape=utt_to_path_shape)

    def intersect(self, lattice: k2.Fsa, use_double_scores=True) -> "Nbest":
        """Intersect this Nbest object with a lattice, get 1-best
        path from the resulting FsaVec, and return a new Nbest object.

        The purpose of this function is to attach scores to an Nbest.

        Args:
          lattice:
            An FsaVec with axes [utt][state][arc]. If it has `aux_labels`, then
            we assume its `labels` are token IDs and `aux_labels` are word IDs.
            If it has only `labels`, we assume its `labels` are word IDs.
          use_double_scores:
            True to use double precision when computing shortest path.
            False to use single precision.
        Returns:
          Return a new Nbest. This new Nbest shares the same shape with `self`,
          while its `fsa` is the 1-best path from intersecting `self.fsa` and
          `lattice`. Also, its `fsa` has non-zero scores and inherits attributes
          for `lattice`.
        """
        # Note: We view each linear FSA as a word sequence
        # and we use the passed lattice to give each word sequence a score.
        #
        # We are not viewing each linear FSAs as a token sequence.
        #
        # So we use k2.invert() here.

        # We use a word fsa to intersect with k2.invert(lattice)
        word_fsa = k2.invert(self.fsa)

        word_fsa.scores.zero_()
        if hasattr(lattice, "aux_labels"):
            # delete token IDs as it is not needed
            del word_fsa.aux_labels
            word_fsa_with_epsilon_loops = k2.linear_fsa_with_self_loops(word_fsa)
        else:
            word_fsa_with_epsilon_loops = k2.linear_fst_with_self_loops(word_fsa)

        path_to_utt_map = self.shape.row_ids(1)

        if hasattr(lattice, "aux_labels"):
            # lattice has token IDs as labels and word IDs as aux_labels.
            # inv_lattice has word IDs as labels and token IDs as aux_labels
            inv_lattice = k2.invert(lattice)
            inv_lattice = k2.arc_sort(inv_lattice)
        else:
            inv_lattice = k2.arc_sort(lattice)

        if inv_lattice.shape[0] == 1:
            path_lattice = _intersect_device(
                inv_lattice,
                word_fsa_with_epsilon_loops,
                b_to_a_map=torch.zeros_like(path_to_utt_map),
                sorted_match_a=True,
            )
        else:
            path_lattice = _intersect_device(
                inv_lattice,
                word_fsa_with_epsilon_loops,
                b_to_a_map=path_to_utt_map,
                sorted_match_a=True,
            )

        # path_lattice has word IDs as labels and token IDs as aux_labels
        # Move to CPU immediately to avoid k2 CUDA crashes
        original_device = path_lattice.device
        logging.info(f"Moving path_lattice to CPU to avoid k2 CUDA issues (original device: {original_device})")
        path_lattice = path_lattice.to('cpu')
        
        # Connect the path_lattice first (on CPU)
        path_lattice = k2.connect(path_lattice)
        
        # Try to process the entire batch at once on CPU
        try:
            path_lattice = k2.top_sort(path_lattice)
            one_best = k2.shortest_path(path_lattice, use_double_scores=use_double_scores)
        except RuntimeError as e:
            # If batch processing fails even on CPU, process individually
            logging.warning(f"Batch top_sort failed on CPU: {e}. Processing FSAs individually.")
            one_best_list = []
            
            for i in range(path_lattice.shape[0]):
                # Extract single FSA
                single_lattice = k2.index_fsa(path_lattice, torch.tensor([i], dtype=torch.int32, device='cpu'))
                
                # Re-connect the single FSA to ensure it's valid
                single_lattice = k2.connect(single_lattice)
                
                # Check if this specific FSA is empty or has issues
                if single_lattice.num_arcs == 0 or single_lattice.shape[0] == 0:
                    logging.warning(f"FSA {i} is empty after connect. Creating dummy path.")
                    # Create a dummy linear FSA
                    labels = torch.tensor([0], dtype=torch.int32, device='cpu')
                    dummy_fsa = k2.linear_fsa(labels, device='cpu')
                    dummy_fsa.scores = torch.tensor([0.0], dtype=torch.float32, device='cpu')
                    if hasattr(lattice, "aux_labels"):
                        dummy_fsa.aux_labels = torch.tensor([0], dtype=torch.int32, device='cpu')
                    one_best_list.append(dummy_fsa)
                else:
                    # Sort and get shortest path for non-empty FSA on CPU
                    try:
                        single_lattice = k2.top_sort(single_lattice)
                        single_best = k2.shortest_path(single_lattice, use_double_scores=use_double_scores)
                        one_best_list.append(single_best)
                    except Exception as e3:
                        logging.warning(f"FSA {i} failed on CPU: {e3}. Creating dummy path.")
                        # Create a dummy linear FSA as fallback
                        labels = torch.tensor([0], dtype=torch.int32, device='cpu')
                        dummy_fsa = k2.linear_fsa(labels, device='cpu')
                        dummy_fsa.scores = torch.tensor([0.0], dtype=torch.float32, device='cpu')
                        if hasattr(lattice, "aux_labels"):
                            dummy_fsa.aux_labels = torch.tensor([0], dtype=torch.int32, device='cpu')
                        dummy_fsa = k2.create_fsa_vec([dummy_fsa])
                        one_best_list.append(dummy_fsa)
            
            # Combine all FSAs using k2.cat
            if len(one_best_list) > 0:
                one_best = k2.cat(one_best_list)
            else:
                # Should not happen, but just in case
                logging.error("No FSAs in one_best_list!")
                labels = torch.tensor([0], dtype=torch.int32, device='cpu')
                dummy_fsa = k2.linear_fsa(labels, device='cpu')
                dummy_fsa.scores = torch.tensor([0.0], dtype=torch.float32, device='cpu')
                one_best = k2.create_fsa_vec([dummy_fsa])
        
        # Move result back to original device
        one_best = one_best.to(original_device)

        one_best = k2.invert(one_best)
        # Now one_best has token IDs as labels and word IDs as aux_labels

        return Nbest(fsa=one_best, shape=self.shape)

    def compute_am_scores(self) -> k2.RaggedTensor:
        """Compute AM scores of each linear FSA (i.e., each path within
        an utterance).

        Hint:
          `self.fsa.scores` contains two parts: acoustic scores (AM scores)
          and n-gram language model scores (LM scores).

        Caution:
          We require that ``self.fsa`` has an attribute ``lm_scores``.

        Returns:
          Return a ragged tensor with 2 axes [utt][path_scores].
          Its dtype is torch.float64.
        """
        scores_shape = self.fsa.arcs.shape().remove_axis(1)
        # scores_shape has axes [path][arc]
        am_scores = self.fsa.scores - self.fsa.lm_scores
        ragged_am_scores = k2.RaggedTensor(scores_shape, am_scores.contiguous())
        tot_scores = ragged_am_scores.sum()

        return k2.RaggedTensor(self.shape, tot_scores)

    def compute_lm_scores(self) -> k2.RaggedTensor:
        """Compute LM scores of each linear FSA (i.e., each path within
        an utterance).

        Hint:
          `self.fsa.scores` contains two parts: acoustic scores (AM scores)
          and n-gram language model scores (LM scores).

        Caution:
          We require that ``self.fsa`` has an attribute ``lm_scores``.

        Returns:
          Return a ragged tensor with 2 axes [utt][path_scores].
          Its dtype is torch.float64.
        """
        scores_shape = self.fsa.arcs.shape().remove_axis(1)
        # scores_shape has axes [path][arc]

        ragged_lm_scores = k2.RaggedTensor(
            scores_shape, self.fsa.lm_scores.contiguous()
        )

        tot_scores = ragged_lm_scores.sum()

        return k2.RaggedTensor(self.shape, tot_scores)

    def tot_scores(self) -> k2.RaggedTensor:
        """Get total scores of FSAs in this Nbest.

        Note:
          Since FSAs in Nbest are just linear FSAs, log-semiring
          and tropical semiring produce the same total scores.

        Returns:
          Return a ragged tensor with two axes [utt][path_scores].
          Its dtype is torch.float64.
        """
        scores_shape = self.fsa.arcs.shape().remove_axis(1)
        # scores_shape has axes [path][arc]

        ragged_scores = k2.RaggedTensor(scores_shape, self.fsa.scores.contiguous())

        tot_scores = ragged_scores.sum()

        return k2.RaggedTensor(self.shape, tot_scores)

    def build_levenshtein_graphs(self) -> k2.Fsa:
        """Return an FsaVec with axes [utt][state][arc]."""
        word_ids = get_texts(self.fsa, return_ragged=True)
        return k2.levenshtein_graph(word_ids)


def nbest_rescore_with_ngram_and_rnn(
    lattice: k2.Fsa,
    ngram_lm: k2.Fsa,
    rnn_lm_model: torch.nn.Module,
    num_paths: int,
    blank_id: int,
    sos_id: int,
    eos_id: int,
    nbest_scale: float = 1.0,
    ngram_lm_scale: float = 0.8,
    rnn_lm_scale: float = 0.5,
    use_double_scores: bool = True,
) -> Dict[str, k2.Fsa]:
    """Extract n-best paths and rescore with both n-gram LM and RNN LM.
    
    This is the complete pipeline for BPE-CTC:
    1. Extract n-best paths from CTC lattice
    2. Compute n-gram LM scores for each path
    3. Compute RNN LM scores for each path
    4. Combine AM + n-gram + RNN scores
    5. Select best path
    
    Args:
      lattice:
        CTC lattice (FsaVec with axes [utt][state][arc]).
        Must have "tokens" attribute.
      ngram_lm:
        Token-level n-gram LM (k2.Fsa). Usually G_4_gram.pt.
      rnn_lm_model:
        RNN language model for rescoring.
      num_paths:
        Number of paths to extract from lattice (e.g., 100).
      blank_id:
        Token ID for blank (usually 0).
      sos_id:
        Token ID for SOS.
      eos_id:
        Token ID for EOS.
      nbest_scale:
        Scale applied to lattice.scores when sampling paths.
      ngram_lm_scale:
        Weight for n-gram LM scores.
      rnn_lm_scale:
        Weight for RNN LM scores.
      use_double_scores:
        Use double precision for computation.
        
    Returns:
      A dict with key like "ngram_scale_X_rnn_scale_Y" and value is
      the best path FSA for each utterance.
    """
    device = lattice.device

    assert len(lattice.shape) == 3
    assert hasattr(lattice, "aux_labels")
    assert hasattr(lattice, "lm_scores")

    assert ngram_lm.shape == (1, None, None)
    assert ngram_lm.device == device

    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        nbest_scale=nbest_scale,
    )
    # nbest.fsa.scores contains 0s

    nbest = nbest.intersect(lattice)

    # Now nbest.fsa has its scores set
    assert hasattr(nbest.fsa, "lm_scores")
    
    # Step 2: Compute AM scores
    print("Computing AM scores...")
    am_scores = nbest.compute_am_scores()
    
    # Step 3: Compute n-gram LM scores
    print("Computing n-gram LM scores...")
    # Invert nbest (labels become aux_labels)
    inv_fsa = k2.invert(nbest.fsa)
    del inv_fsa.aux_labels  # Remove since we don't need them
    inv_fsa.scores.zero_()
    inv_fsa_with_epsilon_loops = k2.linear_fsa_with_self_loops(inv_fsa)
    
    path_to_utt_map = nbest.shape.row_ids(1)
    
    # Ensure ngram_lm is properly sorted for intersection
    # The ngram_lm should already be arc-sorted from decode.py, but we ensure it here
    if not hasattr(ngram_lm, 'properties') or ngram_lm.properties & k2.fsa_properties.ARC_SORTED == 0:
        ngram_lm = k2.arc_sort(ngram_lm)
    
    path_lattice = k2.intersect_device(
        ngram_lm,
        inv_fsa_with_epsilon_loops,
        b_to_a_map=torch.zeros_like(path_to_utt_map),
        sorted_match_a=True,
    )
    
    path_lattice = k2.top_sort(k2.connect(path_lattice))
    one_best = k2.shortest_path(path_lattice, use_double_scores=use_double_scores)
    
    ngram_lm_scores = one_best.get_tot_scores(
        use_double_scores=use_double_scores,
        log_semiring=True,
    )
    # Handle empty paths
    ngram_lm_scores[ngram_lm_scores == float("-inf")] = -1e9
    
    # Step 4: Extract token sequences for RNN LM
    print("Extracting token sequences...")
    tokens_shape = nbest.fsa.arcs.shape().remove_axis(1)
    tokens = k2.RaggedTensor(tokens_shape, nbest.fsa.tokens)
    tokens = tokens.remove_values_leq(0)  # Remove blanks
    token_ids = tokens.tolist()
    
    if len(token_ids) == 0:
        print("Warning: empty token-ids")
        return None
    
    # Step 5: Compute RNN LM scores
    print("Computing RNN LM scores...")
    sos_tokens = add_sos(tokens, sos_id)
    tokens_eos = add_eos(tokens, eos_id)
    sos_tokens_row_splits = sos_tokens.shape.row_splits(1)
    sentence_lengths = sos_tokens_row_splits[1:] - sos_tokens_row_splits[:-1]
    
    x_tokens = sos_tokens.pad(mode="constant", padding_value=blank_id)
    y_tokens = tokens_eos.pad(mode="constant", padding_value=blank_id)
    
    x_tokens = x_tokens.to(torch.int64)
    y_tokens = y_tokens.to(torch.int64)
    sentence_lengths = sentence_lengths.to(torch.int64)
    
    rnn_lm_nll = rnn_lm_model(x=x_tokens, y=y_tokens, lengths=sentence_lengths)
    assert rnn_lm_nll.ndim == 2
    assert rnn_lm_nll.shape[0] == len(token_ids)
    
    rnn_lm_scores = -1 * rnn_lm_nll.sum(dim=1)
    
    # Step 6: Combine all scores
    print(f"Combining scores: AM + {ngram_lm_scale}*n-gram + {rnn_lm_scale}*RNN")
    tot_scores = (
        am_scores.values
        + ngram_lm_scale * ngram_lm_scores
        + rnn_lm_scale * rnn_lm_scores
    )
    
    # Step 7: Select best path
    ragged_tot_scores = k2.RaggedTensor(nbest.shape, tot_scores)
    max_indexes = ragged_tot_scores.argmax()
    best_path = k2.index_fsa(nbest.fsa, max_indexes)
    
    key = f"ngram_scale_{ngram_lm_scale}_rnn_scale_{rnn_lm_scale}"
    
    print(f"Best path selected with key: {key}")
    return {key: best_path}
