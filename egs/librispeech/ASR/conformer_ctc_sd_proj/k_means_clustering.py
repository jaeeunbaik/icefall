"""
K-means clustering for prototype-based knowledge distillation in ASR.

This module provides functions for:
1. Initializing prototypes using K-means clustering on teacher features
2. Computing likelihood/similarity between projection heads and prototypes
3. Monitoring prototype usage and distribution statistics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
import os

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Falling back to sklearn for K-means.")

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available. Some features may not work.")


class PrototypeKMeansManager:
    """
    Manages prototype initialization and operations for knowledge distillation.
    
    This class handles:
    - Prototype initialization using K-means clustering
    - Computing soft assignments between features and prototypes
    - Monitoring prototype usage and distribution statistics
    - Managing per-layer prototype sets
    """
    
    def __init__(
        self,
        target_layers: List[int],
        num_prototypes: int = 256,
        proj_dim: int = 256,
        temperature: float = 2.0,
        use_pca: bool = True,  # Re-enabled for memory-efficient prototypes with projection layer
        pca_dim: int = 128,
        device: torch.device = None,
        save_dir: str = "./prototypes"
    ):
        """
        Initialize the PrototypeKMeansManager.
        
        Args:
            target_layers: List of layer indices to apply prototype matching
            num_prototypes: Number of prototypes per layer (K)
            proj_dim: Dimension of projection head output
            temperature: Temperature for softmax in likelihood computation
            use_pca: Whether to use PCA for dimensionality reduction before K-means
            pca_dim: PCA output dimension
            device: PyTorch device
            save_dir: Directory to save/load prototype files
        """
        self.target_layers = target_layers
        self.num_prototypes = num_prototypes
        self.proj_dim = proj_dim
        self.temperature = temperature
        self.use_pca = use_pca
        self.pca_dim = pca_dim
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        
        # Initialize storage for prototypes
        self.prototypes = {}  # {layer_idx: torch.Tensor}
        self.pca_models = {}  # {layer_idx: PCA}
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        logging.info(f"Initialized PrototypeKMeansManager for layers {target_layers}")
        logging.info(f"K={num_prototypes}, proj_dim={proj_dim}, T={temperature}")
    
    def sample_features_from_teacher(
        self,
        teacher_model: nn.Module,
        dataloader,
        num_samples: int = 500000,
        layer_idx: int = None
    ) -> np.ndarray:
        """
        Sample features from teacher model for prototype initialization.
        
        Args:
            teacher_model: Teacher encoder model
            dataloader: DataLoader for sampling audio data
            num_samples: Target number of feature frames to sample
            layer_idx: Specific layer to extract features from
            
        Returns:
            Sampled features as numpy array [N, feature_dim]
        """
        teacher_model.eval()
        features_list = []
        total_samples = 0
        
        logging.info(f"Sampling {num_samples} features from layer {layer_idx}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if total_samples >= num_samples:
                    break
                
                if 'clean' in batch and 'inputs' in batch['clean']:
                    # Self-distillation format with clean/noisy
                    feature = batch['clean']['inputs'].to(self.device)
                    supervisions = batch['clean']['supervisions']
                    if batch_idx % 100 == 0:  # Reduced logging frequency
                        logging.info(f"DEBUG batch_{batch_idx}: Using clean/noisy format")
                elif 'inputs' in batch:
                    # Standard format
                    feature = batch['inputs'].to(self.device)
                    supervisions = batch['supervisions']
                    if batch_idx % 100 == 0:  # Reduced logging frequency
                        logging.info(f"DEBUG batch_{batch_idx}: Using standard format")
                elif 'feature' in batch:
                    # Legacy format
                    feature = batch['feature'].to(self.device)
                    feature_lens = batch['feature_lens'].to(self.device)
                    if batch_idx % 100 == 0:  # Reduced logging frequency
                        logging.info(f"DEBUG batch_{batch_idx}: Using legacy format")
                else:
                    logging.warning(f"Unknown batch format. Keys: {list(batch.keys())}")
                    continue
                
                # Determine per-utterance feature lengths.
                # If the incoming batch provided "supervisions", it may be in
                # a segment-based format (multiple segments per recording) so
                # we only reuse its 'num_frames' if it matches the batch size.
                if (
                    'supervisions' in locals()
                    and 'num_frames' in supervisions
                    and isinstance(supervisions['num_frames'], (list, tuple, torch.Tensor))
                    and len(supervisions['num_frames']) == feature.size(0)
                ):
                    # Safe to reuse per-utterance lengths
                    if isinstance(supervisions['num_frames'], torch.Tensor):
                        # Removed frequent logging: supervision['num_frames'] is already torch tensor
                        feature_lens = supervisions['num_frames'].to(self.device)
                    else:
                        feature_lens = torch.tensor(supervisions['num_frames'], device=self.device)
                    
                else:
                    # Fallback: use actual feature sequence length for each example
                    feature_lens = torch.full((feature.size(0),), feature.size(1), device=self.device)

                # Create a simple per-utterance supervision dict (one segment per
                # utterance) so encoder_padding_mask can compute lengths without
                # relying on more complex segment structures.
                supervisions = {
                    'sequence_idx': torch.arange(feature.size(0), device=self.device),
                    'start_frame': torch.zeros(feature.size(0), device=self.device),
                    'num_frames': feature_lens.to(self.device),
                    'text': [''] * feature.size(0),  # Empty text for LibriLight
                }
                
                outputs = teacher_model(feature, supervisions)

                if len(outputs) >= 4 and outputs[3] is not None:
                    # Use layer_results if available
                    layer_results = outputs[3]
                    layer_output = layer_results[layer_idx-1]
                                            
                # If model returned a tuple/list (some forwards return multiple
                # values), try to pick the first tensor-like object.
                if not isinstance(layer_output, torch.Tensor):
                    if isinstance(layer_output, (list, tuple)):
                        found = False
                        for item in layer_output:
                            if isinstance(item, torch.Tensor):
                                layer_output = item
                                found = True
                                break
                        if not found:
                            logging.error(f"layer_output is not a tensor and no tensor found inside: {type(layer_output)}")
                            continue  # Skip this batch
                    else:
                        logging.error(f"layer_output is not a tensor: {type(layer_output)}")
                        continue  # Skip this batch
                
                # layer_output should be (T, N, C) format, but might be (N, T, C)
                if layer_output.dim() != 3:
                    logging.error(f"layer_output has wrong dimensions: {layer_output.shape}")
                    continue  # Skip this batch
                
                # Check if we need to transpose from (N, T, C) to (T, N, C)
                if layer_output.size(0) == feature.size(0):  # Batch size matches first dimension
                    layer_output = layer_output.transpose(0, 1)  # (N, T, C) -> (T, N, C)
                
                # Apply feature length masking
                batch_size = layer_output.size(1)
                
                # Estimate number of batches for sampling distribution
                # Check sampler attributes directly without try-except
                sampler = dataloader.sampler
                if hasattr(sampler, 'num_cuts') and sampler.num_cuts is not None:
                    # SimpleCutSampler has num_cuts attribute
                    n_batches = max(1, (sampler.num_cuts + batch_size - 1) // batch_size)
                elif hasattr(sampler, '__len__'):
                    try:
                        sampler_len = len(sampler)
                        n_batches = max(1, (sampler_len + batch_size - 1) // batch_size)
                    except:
                        # Fallback if len() fails
                        n_batches = max(100, num_samples // (batch_size * 10))
                else:
                    # Conservative fallback for even sampling distribution
                    n_batches = max(100, num_samples // (batch_size * 10))

                for b in range(batch_size):
                    # Safe feature length access with bounds checking
                    if b >= len(feature_lens):
                        logging.warning(f"batch index {b} >= feature_lens length {len(feature_lens)}, using last available length")
                        valid_len = feature_lens[-1].item() if len(feature_lens) > 0 else layer_output.size(0)
                    else:
                        valid_len = feature_lens[b].item()
                    
                    # Ensure valid_len doesn't exceed tensor dimensions
                    max_len = layer_output.size(0)
                    valid_len = min(valid_len, max_len)
                    
                    # Safe tensor slicing with bounds checking
                    if valid_len <= 0:
                        logging.warning(f"Invalid valid_len {valid_len}, skipping batch {b}")
                        continue
                        
                    valid_features = layer_output[:valid_len, b]  # [T, D]

                    # Sample random frames from this utterance
                    num_frames = min(valid_len, max(1, num_samples // max(1, n_batches) // max(1, batch_size)))
                    if valid_len > num_frames:
                        indices = torch.randperm(valid_len)[:num_frames]
                        sampled_features = valid_features[indices]
                    else:
                        sampled_features = valid_features
                    
                    features_list.append(sampled_features.cpu().numpy())
                    total_samples += sampled_features.shape[0]
                    
                    if total_samples >= num_samples:
                        break

                # Progress logging (outside try-except block)
                if batch_idx % 100 == 0:
                    logging.info(f"Processed {batch_idx} batches, collected {total_samples} samples")
        
        # Check if we collected any features
        if not features_list:
            raise RuntimeError(f"No features collected for layer {layer_idx}. Check dataloader and model compatibility.")
        
        # Concatenate all features
        features = np.vstack(features_list)
        logging.info(f"Collected {features.shape[0]} feature vectors of dimension {features.shape[1]}")
        
        return features[:num_samples]  # Ensure exact number of samples
    
    def initialize_prototypes(
        self,
        teacher_model: nn.Module,
        dataloader,
        num_samples_per_layer: int = 500000,
        kmeans_iterations: int = 40,
        save_prototypes: bool = True,
        load_if_exists: bool = True
    ) -> Dict[int, torch.Tensor]:
        """
        Initialize prototypes for all target layers using K-means clustering.
        
        Args:
            teacher_model: Teacher encoder model
            dataloader: DataLoader for sampling
            num_samples_per_layer: Number of samples to collect per layer
            kmeans_iterations: Number of K-means iterations
            save_prototypes: Whether to save prototypes to disk
            load_if_exists: Whether to load existing prototypes if available
            
        Returns:
            Dictionary mapping layer indices to prototype tensors
        """
        for layer_idx in self.target_layers:
            prototype_file = os.path.join(self.save_dir, f"prototypes_layer_{layer_idx}.pt")
            pca_file = os.path.join(self.save_dir, f"pca_model_layer_{layer_idx}.pkl")
            
            # Check if prototypes already exist
            if load_if_exists and os.path.exists(prototype_file):
                logging.info(f"Loading existing prototypes for layer {layer_idx}")
                self.prototypes[layer_idx] = torch.load(prototype_file, map_location=self.device)
                
                if self.use_pca and os.path.exists(pca_file):
                    import pickle
                    with open(pca_file, 'rb') as f:
                        self.pca_models[layer_idx] = pickle.load(f)
                continue
            
            # Sample features from teacher
            features = self.sample_features_from_teacher(
                teacher_model, dataloader, num_samples_per_layer, layer_idx
            )
            
            # Apply PCA if requested
            if self.use_pca and features.shape[1] > self.pca_dim:
                logging.info(f"Applying PCA: {features.shape[1]} -> {self.pca_dim}")
                if not SKLEARN_AVAILABLE:
                    logging.warning("sklearn not available. Skipping PCA.")
                    processed_features = features
                else:
                    pca = PCA(n_components=self.pca_dim)
                    processed_features = pca.fit_transform(features)
                    self.pca_models[layer_idx] = pca
                    
                    if save_prototypes:
                        import pickle
                        with open(pca_file, 'wb') as f:
                            pickle.dump(pca, f)
            else:
                processed_features = features
            
            # Perform K-means clustering
            centroids = self.clustering_k_means(
                processed_features, 
                self.num_prototypes, 
                kmeans_iterations,
                layer_idx
            )
            
            # Convert to torch tensor and normalize
            centroids_tensor = torch.from_numpy(centroids).float().to(self.device)
            centroids_tensor = F.normalize(centroids_tensor, dim=1)  # L2 normalize
            
            self.prototypes[layer_idx] = centroids_tensor
            
            # Save prototypes
            if save_prototypes:
                torch.save(centroids_tensor, prototype_file)
                logging.info(f"Saved prototypes for layer {layer_idx} to {prototype_file}")
        
        return self.prototypes
    
    def clustering_k_means(
        self,
        features: np.ndarray,
        k: int,
        n_iterations: int = 40,
        layer_idx: int = None
    ) -> np.ndarray:
        """
        Perform K-means clustering on features to get prototype centroids.
        
        Args:
            features: Feature matrix [N, D]
            k: Number of clusters
            n_iterations: Number of K-means iterations
            layer_idx: Layer index for logging
            
        Returns:
            Cluster centroids [K, D]
        """
        logging.info(f"Running K-means clustering for layer {layer_idx}: {features.shape} -> {k} clusters")
        
        # Normalize features
        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        
        if FAISS_AVAILABLE and features.shape[0] > 10000:
            # Use FAISS for large datasets
            logging.info("Using FAISS K-means")
            d = features.shape[1]
            kmeans = faiss.Kmeans(d, k, niter=n_iterations, gpu=torch.cuda.is_available())
            kmeans.train(features.astype(np.float32))
            centroids = kmeans.centroids
        elif SKLEARN_AVAILABLE:
            # Use sklearn K-means
            logging.info("Using sklearn K-means")
            kmeans = KMeans(n_clusters=k, max_iter=n_iterations, random_state=42, n_init=1)
            kmeans.fit(features)
            centroids = kmeans.cluster_centers_
        else:
            # Simple numpy implementation
            logging.info("Using numpy K-means implementation")
            centroids = self._numpy_kmeans(features, k, n_iterations)
        
        # Normalize centroids
        centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
        
        logging.info(f"K-means completed. Centroids shape: {centroids.shape}")
        return centroids
    
    def _numpy_kmeans(self, features: np.ndarray, k: int, n_iterations: int) -> np.ndarray:
        """Simple numpy implementation of K-means."""
        n_samples, n_features = features.shape
        
        # Initialize centroids randomly
        centroids = features[np.random.choice(n_samples, k, replace=False)]
        
        for i in range(n_iterations):
            # Assign points to closest centroid
            distances = np.sqrt(((features - centroids[:, np.newaxis])**2).sum(axis=2))
            closest_centroid = np.argmin(distances, axis=0)
            
            # Update centroids
            for j in range(k):
                if np.sum(closest_centroid == j) > 0:
                    centroids[j] = features[closest_centroid == j].mean(axis=0)
        
        return centroids
    
    def compute_likelihood(
        self,
        teacher_features: torch.Tensor,
        student_features: torch.Tensor,
        layer_idx: int,
        frame_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute likelihood/similarity between projection features and prototypes.
        
        Args:
            teacher_features: Teacher projection output [B, T, D]
            student_features: Student projection output [B, T, D]
            layer_idx: Layer index to get corresponding prototypes
            frame_mask: Mask for valid frames [B, T], 1 for valid frames
            
        Returns:
            Dictionary containing:
                - kl_loss: KL divergence loss
                - teacher_probs: Teacher soft assignments [B, T, K]
                - student_probs: Student soft assignments [B, T, K]
                - logits_teacher: Teacher logits [B, T, K]
                - logits_student: Student logits [B, T, K]
        """
        if layer_idx not in self.prototypes:
            raise ValueError(f"Prototypes for layer {layer_idx} not initialized")
        
        prototypes = self.prototypes[layer_idx]  # [K, D]
        
        # L2 normalize features
        teacher_features = F.normalize(teacher_features, dim=-1)  # [B, T, D]
        student_features = F.normalize(student_features, dim=-1)  # [B, T, D]
        
        # Compute cosine similarity logits
        logits_teacher = torch.einsum('btd,kd->btk', teacher_features, prototypes) / self.temperature
        logits_student = torch.einsum('btd,kd->btk', student_features, prototypes) / self.temperature
        
        # Compute probabilities
        log_prob_teacher = F.log_softmax(logits_teacher, dim=-1)
        log_prob_student = F.log_softmax(logits_student, dim=-1)
        prob_teacher = log_prob_teacher.exp()
        prob_student = log_prob_student.exp()
        
        # Compute KL divergence: KL(teacher || student)
        kl_per_frame = torch.sum(prob_teacher * (log_prob_teacher - log_prob_student), dim=-1)  # [B, T]
        
        # Apply frame mask if provided
        if frame_mask is not None:
            kl_per_frame = kl_per_frame * frame_mask
            valid_frames = frame_mask.sum()
            kl_loss = kl_per_frame.sum() / (valid_frames + 1e-8)
        else:
            kl_loss = kl_per_frame.mean()
        
        # Scale by temperature squared (knowledge distillation style)
        kl_loss = kl_loss * (self.temperature ** 2)
        
        return {
            'kl_loss': kl_loss,
            'teacher_probs': prob_teacher,
            'student_probs': prob_student,
            'logits_teacher': logits_teacher,
            'logits_student': logits_student
        }
    
    def monitor_prototype_usage(
        self,
        teacher_probs: torch.Tensor,
        student_probs: torch.Tensor,
        layer_idx: int,
        frame_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Monitor prototype usage statistics for debugging and analysis.
        
        Args:
            teacher_probs: Teacher soft assignments [B, T, K]
            student_probs: Student soft assignments [B, T, K]
            layer_idx: Layer index
            frame_mask: Mask for valid frames [B, T]
            
        Returns:
            Dictionary with monitoring statistics
        """
        K = teacher_probs.size(-1)
        
        # Apply mask if provided
        if frame_mask is not None:
            # Expand mask to [B, T, 1] for broadcasting
            mask_expanded = frame_mask.unsqueeze(-1)
            teacher_probs_masked = teacher_probs * mask_expanded
            student_probs_masked = student_probs * mask_expanded
            valid_frames = frame_mask.sum().item()
        else:
            teacher_probs_masked = teacher_probs
            student_probs_masked = student_probs
            valid_frames = teacher_probs.numel() // K
        
        # Compute entropy
        eps = 1e-8
        entropy_teacher = -torch.sum(teacher_probs_masked * torch.log(teacher_probs_masked + eps), dim=-1)
        entropy_student = -torch.sum(student_probs_masked * torch.log(student_probs_masked + eps), dim=-1)
        
        if frame_mask is not None:
            entropy_teacher = (entropy_teacher * frame_mask).sum() / (frame_mask.sum() + eps)
            entropy_student = (entropy_student * frame_mask).sum() / (frame_mask.sum() + eps)
        else:
            entropy_teacher = entropy_teacher.mean()
            entropy_student = entropy_student.mean()
        
        # Normalized entropy
        max_entropy = np.log(K)
        norm_entropy_teacher = entropy_teacher.item() / max_entropy
        norm_entropy_student = entropy_student.item() / max_entropy
        
        # Prototype usage statistics (using argmax for hard assignment)
        teacher_assignments = torch.argmax(teacher_probs, dim=-1)  # [B, T]
        student_assignments = torch.argmax(student_probs, dim=-1)  # [B, T]
        
        if frame_mask is not None:
            teacher_assignments = teacher_assignments[frame_mask.bool()]
            student_assignments = student_assignments[frame_mask.bool()]
        
        teacher_assignments = teacher_assignments.view(-1).cpu().numpy()
        student_assignments = student_assignments.view(-1).cpu().numpy()
        
        # Count usage
        teacher_counts = Counter(teacher_assignments)
        student_counts = Counter(student_assignments)
        
        teacher_usage = np.zeros(K)
        student_usage = np.zeros(K)
        
        for k, count in teacher_counts.items():
            teacher_usage[k] = count
        for k, count in student_counts.items():
            student_usage[k] = count
        
        # Compute statistics
        teacher_unused_frac = (teacher_usage == 0).sum() / K
        student_unused_frac = (student_usage == 0).sum() / K
        
        # Top-k usage share
        topk = max(1, int(0.1 * K))
        teacher_topk_share = np.sort(teacher_usage)[-topk:].sum() / (teacher_usage.sum() + eps)
        student_topk_share = np.sort(student_usage)[-topk:].sum() / (student_usage.sum() + eps)
        
        # Teacher-student feature similarity
        teacher_flat = teacher_probs.view(-1, K)
        student_flat = student_probs.view(-1, K)
        
        if frame_mask is not None:
            mask_flat = frame_mask.view(-1)
            teacher_flat = teacher_flat[mask_flat.bool()]
            student_flat = student_flat[mask_flat.bool()]
        
        cosine_sim = F.cosine_similarity(teacher_flat, student_flat, dim=1).mean().item()
        
        return {
            f'layer_{layer_idx}/norm_entropy_teacher': norm_entropy_teacher,
            f'layer_{layer_idx}/norm_entropy_student': norm_entropy_student,
            f'layer_{layer_idx}/unused_frac_teacher': teacher_unused_frac,
            f'layer_{layer_idx}/unused_frac_student': student_unused_frac,
            f'layer_{layer_idx}/top10_share_teacher': teacher_topk_share,
            f'layer_{layer_idx}/top10_share_student': student_topk_share,
            f'layer_{layer_idx}/cosine_similarity': cosine_sim,
            f'layer_{layer_idx}/valid_frames': valid_frames
        }
    
    def get_prototype_tensor(self, layer_idx: int) -> torch.Tensor:
        """Get prototype tensor for a specific layer."""
        if layer_idx not in self.prototypes:
            raise ValueError(f"Prototypes for layer {layer_idx} not initialized")
        return self.prototypes[layer_idx]
    
    def update_prototypes_ema(
        self,
        layer_idx: int,
        features: torch.Tensor,
        assignments: torch.Tensor,
        momentum: float = 0.99,
        frame_mask: Optional[torch.Tensor] = None
    ):
        """
        Update prototypes using exponential moving average.
        
        Args:
            layer_idx: Layer index
            features: Current batch features [B, T, D]
            assignments: Soft assignments [B, T, K]
            momentum: EMA momentum
            frame_mask: Mask for valid frames [B, T]
        """
        if layer_idx not in self.prototypes:
            raise ValueError(f"Prototypes for layer {layer_idx} not initialized")
        
        prototypes = self.prototypes[layer_idx]  # [K, D]
        
        # Apply mask if provided
        if frame_mask is not None:
            mask_expanded = frame_mask.unsqueeze(-1)  # [B, T, 1]
            features = features * mask_expanded
            assignments = assignments * mask_expanded
        
        # Compute weighted average features for each prototype
        # assignments: [B, T, K], features: [B, T, D]
        assignment_sums = assignments.sum(dim=(0, 1))  # [K]
        weighted_features = torch.einsum('btk,btd->kd', assignments, features)  # [K, D]
        
        # Update prototypes where there are sufficient assignments
        for k in range(self.num_prototypes):
            if assignment_sums[k] > 1e-8:  # Avoid division by zero
                new_centroid = weighted_features[k] / assignment_sums[k]
                new_centroid = F.normalize(new_centroid, dim=0)
                prototypes[k] = momentum * prototypes[k] + (1 - momentum) * new_centroid
        
        # Re-normalize all prototypes
        self.prototypes[layer_idx] = F.normalize(prototypes, dim=1)
    
    def save_prototypes(self, save_dir: str = None):
        """Save all prototypes to disk."""
        save_dir = save_dir or self.save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        for layer_idx, prototypes in self.prototypes.items():
            save_path = os.path.join(save_dir, f"prototypes_layer_{layer_idx}.pt")
            torch.save(prototypes.cpu(), save_path)
            logging.info(f"Saved prototypes for layer {layer_idx} to {save_path}")
    
    def load_prototypes(self, load_dir: str = None):
        """Load prototypes from disk."""
        load_dir = load_dir or self.save_dir
        
        for layer_idx in self.target_layers:
            load_path = os.path.join(load_dir, f"prototypes_layer_{layer_idx}.pt")
            if os.path.exists(load_path):
                self.prototypes[layer_idx] = torch.load(load_path, map_location=self.device)
                logging.info(f"Loaded prototypes for layer {layer_idx} from {load_path}")
            else:
                logging.warning(f"Prototype file not found: {load_path}")


def create_prototype_manager(
    target_layers: List[int],
    num_prototypes: int = 256,
    proj_dim: int = 256,
    temperature: float = 2.0,
    save_dir: str = "./prototypes"
) -> PrototypeKMeansManager:
    """
    Factory function to create a PrototypeKMeansManager instance.
    
    Args:
        target_layers: List of layer indices for prototype matching
        num_prototypes: Number of prototypes per layer
        proj_dim: Projection dimension
        temperature: Softmax temperature
        save_dir: Directory for saving prototypes
        
    Returns:
        Configured PrototypeKMeansManager instance
    """
    return PrototypeKMeansManager(
        target_layers=target_layers,
        num_prototypes=num_prototypes,
        proj_dim=proj_dim,
        temperature=temperature,
        save_dir=save_dir
    )
