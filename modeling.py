
"""Advanced model architectures for BirdCLEF 2026 — targeting 0.95+.

Architecture overview:
  1. AudioEncoder: Perch embeddings → projection
  2. TemporalTransformer: Self-attention over time frames
  3. HierarchicalHead: Coarse(5-class) + Fine(234-species) prediction
  4. GeoEncoder: Location encoding as auxiliary feature
  5. BirdCLEFModel: Full ensemble-ready model
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# Positional Encoding
# ============================================================================

class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding for temporal frames."""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]


# ============================================================================
# Audio Encoder (Projection + optional fine-tuning interface)
# ============================================================================

class AudioEncoder(nn.Module):
    """
    Projects Perch embeddings to model dimension.
    
    For phase 1: frozen Perch → projection MLP
    For phase 2: replace with fine-tuned Perch wrapper
    """
    def __init__(self, in_dim=1280, d_model=512, dropout=0.2, n_layers=2):
        super().__init__()
        layers = []
        # First layer: in_dim -> d_model*2
        layers += [
            nn.Linear(in_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        # Middle layers: d_model*2 -> d_model*2 (if n_layers > 2)
        for _ in range(n_layers - 2):
            layers += [
                nn.Linear(d_model * 2, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        # Final layer: d_model*2 -> d_model
        layers += [
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        ]
        self.projection = nn.Sequential(*layers)
        self.d_model = d_model
    
    def forward(self, x):
        """
        x: (B, in_dim) or (B, T, in_dim)
        returns: (B, d_model) or (B, T, d_model)
        """
        return self.projection(x)


# ============================================================================
# Multi-Scale Temporal Transformer
# ============================================================================

class TemporalTransformer(nn.Module):
    """
    Self-attention over temporal frames.
    
    For soundscape segments, Perch embeddings can be extracted at multiple
    time offsets within a 5s window, producing a sequence of embeddings.
    This transformer learns which temporal positions carry the signal.
    
    Architecture:
      - Learned positional encoding
      - N transformer encoder layers with pre-norm
      - Attention pooling → fixed-dim output
    """
    def __init__(self, d_model=512, n_heads=8, n_layers=4, 
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.pos_encoder = LearnedPositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Learnable CLS token for global pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Attention pooling as alternative
        self.attn_pool = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        x: (B, T, d_model) — temporal sequence of projected embeddings
        mask: (B, T) — optional padding mask (True = valid)
        returns: (B, d_model) — aggregated representation
        """
        B, T, D = x.shape
        
        # Add positional encoding and CLS token
        x = self.pos_encoder(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+T, D)
        
        # Extend mask for CLS token
        if mask is not None:
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
            mask = torch.cat([cls_mask, mask], dim=1)
        
        # Transformer encoding
        src_key_padding_mask = ~mask if mask is not None else None
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Attention pooling (alternative to just using CLS output)
        pooled, _ = self.attn_pool(
            self.pool_query.expand(B, -1, -1),
            x, x,
        )
        
        return self.norm(self.dropout(pooled.squeeze(1)))


# ============================================================================
# Hierarchical Classification Head
# ============================================================================

class HierarchicalHead(nn.Module):
    """
    Two-level hierarchical classification:
      Level 1: Coarse class (Aves, Amphibia, Insecta, Mammalia, Reptilia) — 5 classes
      Level 2: Fine species within each coarse class — 234 total
    
    The hierarchical loss encourages the model to:
      a) Correctly identify the coarse taxonomic class
      b) Then distinguish species within that class
    
    This is especially helpful for:
      - Rare species (share features with common species in same class)
      - Insect sonotypes (benefit from shared insect-level features)
      - The 28 unseen species (can at least predict coarse class from similarity)
    """
    def __init__(self, d_model=512, n_species=234, 
                 coarse_map=None,  # species_idx → coarse_idx
                 num_coarse=5,
                 dropout=0.2):
        super().__init__()
        self.n_species = n_species
        self.num_coarse = num_coarse
        
        # Coarse classifier
        self.coarse_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_coarse),
        )
        
        # Fine classifiers — one per coarse class
        # Species within each coarse class share a dedicated head
        self.fine_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_per_coarse),
            )
            for n_per_coarse in self._get_species_per_coarse(coarse_map, n_species, num_coarse)
        ])
        
        # Store species-to-coarse mapping for loss computation
        if coarse_map is not None:
            self.register_buffer('coarse_map', torch.tensor(coarse_map))
        else:
            self.register_buffer('coarse_map', torch.zeros(n_species, dtype=torch.long))
        
        # Learnable species embeddings for similarity-based prediction
        self.species_embeddings = nn.Parameter(
            torch.randn(n_species, d_model) * 0.02
        )
    
    def _get_species_per_coarse(self, coarse_map, n_species, num_coarse):
        if coarse_map is None:
            base = n_species // num_coarse
            rem = n_species % num_coarse
            return [base + 1] * rem + [base] * (num_coarse - rem)
        counts = [0] * num_coarse
        for c in coarse_map:
            counts[c] += 1
        return counts
    
    def forward(self, x):
        """
        x: (B, d_model)
        returns:
          coarse_logits: (B, 5)
          fine_logits: (B, 234)
          similarity_logits: (B, 234) — cosine similarity to species embeddings
        """
        # Coarse prediction
        coarse_logits = self.coarse_head(x)
        
        # Fine predictions from each head
        fine_outputs = []
        for head in self.fine_heads:
            fine_outputs.append(head(x))
        fine_logits = torch.cat(fine_outputs, dim=1)
        
        # Similarity-based prediction (ensures unseen species get non-zero prob)
        x_norm = F.normalize(x, dim=1)
        sp_norm = F.normalize(self.species_embeddings, dim=1)
        similarity_logits = x_norm @ sp_norm.T * 10  # temperature scaling
        
        return coarse_logits, fine_logits, similarity_logits


# ============================================================================
# Geographic Encoder
# ============================================================================

class GeoEncoder(nn.Module):
    """
    Encode geographic coordinates as additional features.
    Species distributions have strong geographic structure.
    """
    def __init__(self, d_model=512, n_frequencies=32):
        super().__init__()
        self.n_freq = n_frequencies
        self.d_model = d_model
        
        # Project Fourier features to d_model
        self.projection = nn.Sequential(
            nn.Linear(n_frequencies * 4, d_model),  # 4 = 2 coords * (sin + cos)
            nn.GELU(),
            nn.Linear(d_model, d_model // 4),
        )
    
    def _fourier_features(self, coords):
        """Encode (lat, lon) with sin/cos at multiple frequencies."""
        B = coords.shape[0]
        freqs = 2.0 ** torch.linspace(0, 8, self.n_freq, device=coords.device)
        features = []
        for coord in [coords[:, 0:1], coords[:, 1:2]]:
            for freq in freqs:
                features.append(torch.sin(coord * freq))
                features.append(torch.cos(coord * freq))
        return torch.cat(features, dim=1)
    
    def forward(self, lat, lon):
        coords = torch.stack([lat, lon], dim=1)
        fourier = self._fourier_features(coords)
        return self.projection(fourier)


# ============================================================================
# Full BirdCLEF Model
# ============================================================================

class BirdCLEFModel(nn.Module):
    """
    Complete model for BirdCLEF 2026.
    
    Combines:
      - AudioEncoder (Perch projection)
      - TemporalTransformer (optional, for multi-frame input)
      - HierarchicalHead (coarse + fine classification)
      - GeoEncoder (location auxiliary)
    """
    def __init__(self, 
                 in_dim=1280,
                 d_model=512,
                 n_species=234,
                 n_coarse=5,
                 coarse_map=None,
                 n_transformer_layers=4,
                 n_heads=8,
                 dropout=0.2,
                 use_temporal=False,
                 use_geo=False,
                 n_audio_layers=2):
        super().__init__()
        self.use_temporal = use_temporal
        self.use_geo = use_geo
        self.d_model = d_model
        
        # Audio projection
        self.audio_encoder = AudioEncoder(in_dim, d_model, dropout, n_audio_layers)
        
        # Temporal transformer (for multi-frame inputs)
        if use_temporal:
            self.temporal_transformer = TemporalTransformer(
                d_model, n_heads, n_transformer_layers, d_model * 4, dropout
            )
        
        # Hierarchical classifier
        self.head = HierarchicalHead(d_model, n_species, coarse_map, n_coarse, dropout)
        
        # Geographic encoder
        if use_geo:
            self.geo_encoder = GeoEncoder(d_model)
            self.geo_fusion = nn.Sequential(
                nn.Linear(d_model + d_model // 4, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
        
        # Simple species presence prior (learned bias)
        self.species_bias = nn.Parameter(torch.zeros(n_species))
    
    def forward(self, x, lat=None, lon=None, mask=None):
        """
        x: (B, in_dim) for single-frame or (B, T, in_dim) for multi-frame
        lat, lon: (B,) optional geographic coordinates
        mask: (B, T) optional temporal mask
        
        returns: dict with 'coarse', 'fine', 'similarity', 'logits'
        """
        B = x.shape[0]
        
        if self.use_temporal and x.dim() == 3:
            # Multi-frame input
            x = self.audio_encoder(x)  # (B, T, d_model)
            x = self.temporal_transformer(x, mask)  # (B, d_model)
        else:
            # Single-frame input
            if x.dim() == 3:
                x = x.squeeze(1)
            x = self.audio_encoder(x)  # (B, d_model)
        
        # Fuse with geographic features
        if self.use_geo and lat is not None and lon is not None:
            geo_feat = self.geo_encoder(lat, lon)
            x = self.geo_fusion(torch.cat([x, geo_feat], dim=1))
        
        # Hierarchical prediction
        coarse_logits, fine_logits, similarity_logits = self.head(x)
        
        # Combine predictions
        # Final logits = fine + similarity * coarse_prior
        coarse_probs = F.softmax(coarse_logits, dim=1)
        
        # Map coarse probabilities to per-species prior
        # This boosts species whose coarse class is predicted with high confidence
        species_boost = coarse_probs[:, self.head.coarse_map] * 0.1
        
        logits = fine_logits + similarity_logits * 0.3 + self.species_bias + species_boost
        
        return {
            'logits': logits,
            'coarse_logits': coarse_logits,
            'fine_logits': fine_logits,
            'similarity_logits': similarity_logits,
            'species_bias': self.species_bias,
        }


# ============================================================================
# Simple MLP Baseline (for comparison)
# ============================================================================

class MLPBaseline(nn.Module):
    """Original MLP probe — kept for baseline comparison."""
    def __init__(self, in_dim=1280, n_classes=234, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = [2048, 1024, 512]
        layers = []
        prev = in_dim
        for h in hidden:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            ])
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return {'logits': self.net(x)}


# ============================================================================
# Few-Shot Prototypical Head for Unseen Species
# ============================================================================

class PrototypicalHead(nn.Module):
    """
    Few-shot prototypical network for handling 28 unseen species.
    
    During training: learn a metric space where species embeddings cluster.
    During inference: unseen species can be detected by similarity to 
    a small number of manually labeled examples.
    """
    def __init__(self, d_model=512, n_species=234):
        super().__init__()
        self.prototype_embeddings = nn.Parameter(
            torch.randn(n_species, d_model) * 0.02
        )
        self.temperature = nn.Parameter(torch.tensor(10.0))
    
    def forward(self, x, support_embeddings=None, support_labels=None):
        """
        x: (B, d_model) — query embeddings
        support_embeddings: (N, d_model) — few-shot support examples
        support_labels: (N,) — support labels
        
        Returns logits based on cosine similarity to prototypes.
        """
        if support_embeddings is not None and support_labels is not None:
            # Few-shot: compute prototypes from support set
            prototypes = torch.zeros_like(self.prototype_embeddings)
            counts = torch.zeros(self.prototype_embeddings.shape[0], device=x.device)
            for i, label in enumerate(support_labels):
                prototypes[label] += support_embeddings[i]
                counts[label] += 1
            # Average and fill missing with learned embeddings
            mask = counts > 0
            prototypes[mask] = prototypes[mask] / counts[mask].unsqueeze(1)
            prototypes[~mask] = self.prototype_embeddings[~mask]
        else:
            prototypes = self.prototype_embeddings
        
        # Cosine similarity
        x_norm = F.normalize(x, dim=1)
        p_norm = F.normalize(prototypes, dim=1)
        return x_norm @ p_norm.T * self.temperature
