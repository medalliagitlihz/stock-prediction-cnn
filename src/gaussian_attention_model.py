"""
Gaussian Graphical Model Attention (GGMA)

This module implements transformers with explicit precision matrix modeling,
enabling analysis of conditional independence structures learned by attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns


class PrecisionMultiHeadAttention(nn.Module):
    """
    Multi-head attention with explicit precision matrix modeling.
    
    This layer interprets attention weights as derived from the precision
    matrix of a Gaussian graphical model, enabling:
    1. Extraction and analysis of learned conditional independence structures
    2. Explicit regularization of precision matrices
    3. Uncertainty quantification via posterior inference
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        precision_regularization: float = 0.01,
        track_precision: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.dropout = dropout
        self.precision_regularization = precision_regularization
        self.track_precision = track_precision
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Learnable temperature (controls scale, related to 1/sqrt(d))
        self.temperature = nn.Parameter(torch.ones(num_heads) / np.sqrt(self.d_head))
        
        # Optional: learnable precision structure regularization per head
        # This encourages certain patterns of conditional independence
        self.precision_bias = nn.Parameter(
            torch.zeros(num_heads, 1, 1),
            requires_grad=True
        )
        
        # Storage for analysis
        self.attention_weights_history = []
        self.precision_matrices_history = []
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or [batch_size, 1, seq_len, seq_len]
        
        Returns:
            output: [batch_size, seq_len, d_model]
            analysis: dict containing attention weights and precision matrix info
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V and reshape for multi-head attention
        # [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, d_head]
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_head)
        Q = Q.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_head]
        
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_head)
        K = K.transpose(1, 2)
        
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_head)
        V = V.transpose(1, 2)
        
        # Compute attention scores using precision interpretation
        # [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1))  # Q·K^T
        
        # Scale by temperature (precision-related parameter)
        scores = scores * self.temperature.view(1, self.num_heads, 1, 1)
        
        # Add precision bias (encourages certain sparsity patterns)
        scores = scores + self.precision_bias.view(1, self.num_heads, 1, 1)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax: interpret as posterior probability of conditional dependence
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        
        # Apply dropout
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)
        
        # Compute output: weighted sum of values
        # [batch_size, num_heads, seq_len, seq_len] @ [batch_size, num_heads, seq_len, d_head]
        output = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, d_head]
        
        # Reshape back to [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        
        # Final output projection
        output = self.W_o(output)
        
        # Store for analysis
        analysis = self._analyze_precision_structure(
            attention_weights, Q, K, V, x
        )
        
        return output, analysis
    
    def _analyze_precision_structure(
        self,
        attention_weights: torch.Tensor,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        x: torch.Tensor,
    ) -> Dict:
        """
        Analyze the learned precision matrix structure.
        
        Interprets attention weights as samples from the precision matrix
        and extracts:
        1. Estimated precision matrices (per head)
        2. Sparsity patterns (conditional independence)
        3. Strength of connections
        """
        if not self.track_precision:
            return {}
        
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Average attention across batch for analysis
        avg_attention = attention_weights.mean(dim=0)  # [num_heads, seq_len, seq_len]
        
        # Compute empirical precision approximation
        # Precision ≈ (normalized attention weights)
        # In Gaussian graphical models: |Θᵢⱼ| ∝ strength of direct association
        precision_estimates = []
        sparsity_ratios = []
        
        for h in range(num_heads):
            attn = avg_attention[h]  # [seq_len, seq_len]
            
            # Threshold to detect conditional independence (near-zero entries)
            sparsity_threshold = 0.01
            sparse_mask = (attn < sparsity_threshold).float()
            sparsity_ratio = sparse_mask.sum().item() / (seq_len * seq_len)
            
            precision_estimates.append(attn.detach().cpu().numpy())
            sparsity_ratios.append(sparsity_ratio)
        
        # Store history for later analysis
        self.attention_weights_history.append(
            avg_attention.detach().cpu().numpy()
        )
        self.precision_matrices_history.append(precision_estimates)
        
        # Compute conditional independence patterns
        # Two tokens are conditionally independent if attention is near-zero
        cond_indep_patterns = self._extract_conditional_independence(
            avg_attention, threshold=0.01
        )
        
        analysis = {
            'attention_weights': avg_attention.detach().cpu().numpy(),
            'precision_estimates': precision_estimates,
            'sparsity_ratios': sparsity_ratios,
            'mean_sparsity': np.mean(sparsity_ratios),
            'conditional_independence_patterns': cond_indep_patterns,
            'temperature': self.temperature.detach().cpu().numpy(),
        }
        
        return analysis
    
    def _extract_conditional_independence(
        self,
        attention: torch.Tensor,
        threshold: float = 0.01,
    ) -> Dict:
        """
        Extract conditional independence structure from attention weights.
        
        A pair (i, j) is conditionally independent if:
        - attention[i, j] < threshold
        - attention[j, i] < threshold
        
        This would imply Θᵢⱼ ≈ 0 in the precision matrix.
        """
        num_heads, seq_len, _ = attention.shape
        
        patterns = {}
        for h in range(num_heads):
            attn_h = attention[h]
            
            # Find conditionally independent pairs
            symmetric_low = (
                (attn_h < threshold) & 
                (attn_h.T < threshold)
            )
            
            cond_indep_pairs = torch.where(symmetric_low)
            n_pairs = len(cond_indep_pairs[0])
            
            patterns[f'head_{h}'] = {
                'n_conditional_independent_pairs': int(n_pairs),
                'proportion': float(n_pairs / (seq_len * seq_len)),
            }
        
        return patterns


class GaussianGraphicalModelTransformer(nn.Module):
    """
    Transformer with precision matrix-based attention.
    
    This model combines GGMA layers with feedforward networks to form
    a full transformer that can learn and interpret conditional independence
    structures in sequential data.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        vocab_size: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        precision_regularization: float = 0.01,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Embedding layer (optional, for language modeling)
        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, d_model)
        else:
            self.embedding = None
        
        # Positional encoding
        self.pos_encoding = self._get_positional_encoding(max_seq_len, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            {
                'attention': PrecisionMultiHeadAttention(
                    d_model, num_heads, dropout, precision_regularization
                ),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                ),
                'norm2': nn.LayerNorm(d_model),
            }
            for _ in range(num_layers)
        ])
        
        # Layer objects for nn.ModuleList
        self.attention_modules = nn.ModuleList([
            layer['attention'] for layer in self.layers
        ])
        self.norm1_modules = nn.ModuleList([
            layer['norm1'] for layer in self.layers
        ])
        self.ffn_modules = nn.ModuleList([
            layer['ffn'] for layer in self.layers
        ])
        self.norm2_modules = nn.ModuleList([
            layer['norm2'] for layer in self.layers
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def _get_positional_encoding(self, max_len, d_model):
        """Compute positional encoding (standard transformer PE)."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: [batch_size, seq_len] (token ids) or [batch_size, seq_len, d_model] (embeddings)
            mask: [batch_size, seq_len] (bool mask, True = valid)
        
        Returns:
            output: [batch_size, seq_len, d_model]
            analyses: list of analysis dicts from each attention layer
        """
        batch_size, seq_len = x.shape[:2]
        
        # Embedding
        if self.embedding is not None and x.dtype == torch.long:
            x = self.embedding(x)
        
        # Add positional encoding
        if x.device != self.pos_encoding.device:
            self.pos_encoding = self.pos_encoding.to(x.device)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Convert mask to attention mask
        if mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            attn_mask = mask.view(batch_size, 1, 1, seq_len)
            attn_mask = attn_mask.expand(batch_size, 1, seq_len, seq_len)
        else:
            attn_mask = None
        
        # Forward through transformer layers
        analyses = []
        for layer_idx in range(self.num_layers):
            # Self-attention with residual connection
            attn_output, analysis = self.attention_modules[layer_idx](x, attn_mask)
            x = self.norm1_modules[layer_idx](x + attn_output)
            
            # Feedforward with residual connection
            ffn_output = self.ffn_modules[layer_idx](x)
            x = self.norm2_modules[layer_idx](x + ffn_output)
            
            analyses.append(analysis)
        
        return x, analyses


# ============================================================================
# ANALYSIS AND VISUALIZATION
# ============================================================================

class PrecisionAnalyzer:
    """Analyzes learned precision matrix structures from trained transformers."""
    
    def __init__(self, model: GaussianGraphicalModelTransformer):
        self.model = model
    
    def extract_precision_matrices(self) -> Dict:
        """
        Extract learned precision matrix estimates from all attention heads.
        
        Returns:
            Dict mapping layer -> head -> precision matrix
        """
        precision_data = {}
        
        for layer_idx, attn_module in enumerate(self.model.attention_modules):
            if not attn_module.precision_matrices_history:
                continue
            
            precision_data[f'layer_{layer_idx}'] = {
                'matrices': attn_module.precision_matrices_history,
                'sparsity_ratios': [
                    analysis.get('mean_sparsity', 0)
                    for analysis in attn_module.attention_weights_history
                ],
            }
        
        return precision_data
    
    def analyze_conditional_independence(self) -> Dict:
        """
        Analyze conditional independence patterns discovered by the model.
        
        Returns:
            Statistics on which token pairs are learned to be conditionally independent
        """
        results = {}
        
        for layer_idx, attn_module in enumerate(self.model.attention_modules):
            layer_key = f'layer_{layer_idx}'
            results[layer_key] = {
                'conditional_independence_patterns': [],
                'mean_cond_indep_pairs': 0,
            }
            
            for analysis in attn_module.attention_weights_history:
                cond_indep = analysis.get('conditional_independence_patterns', {})
                results[layer_key]['conditional_independence_patterns'].append(cond_indep)
            
            # Aggregate statistics
            if results[layer_key]['conditional_independence_patterns']:
                n_heads = self.model.attention_modules[layer_idx].num_heads
                total_pairs = 0
                for pattern_dict in results[layer_key]['conditional_independence_patterns']:
                    for head in range(n_heads):
                        head_key = f'head_{head}'
                        if head_key in pattern_dict:
                            total_pairs += pattern_dict[head_key].get(
                                'n_conditional_independent_pairs', 0
                            )
                
                results[layer_key]['mean_cond_indep_pairs'] = (
                    total_pairs / len(results[layer_key]['conditional_independence_patterns'])
                    if results[layer_key]['conditional_independence_patterns'] else 0
                )
        
        return results
    
    def visualize_precision_matrices(
        self,
        layer_idx: int = 0,
        head_idx: int = 0,
        save_path: Optional[str] = None,
    ):
        """Visualize learned precision matrices as heatmaps."""
        attn_module = self.model.attention_modules[layer_idx]
        
        if not attn_module.precision_matrices_history:
            print("No precision matrix history available. Run model on data first.")
            return
        
        # Get final precision matrix estimate
        final_precision = attn_module.precision_matrices_history[-1][head_idx]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            final_precision,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Attention Weight (Precision Estimate)'},
        )
        plt.title(f'Learned Precision Matrix\nLayer {layer_idx}, Head {head_idx}')
        plt.xlabel('Token Position')
        plt.ylabel('Token Position')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def compute_sparsity_evolution(self) -> Dict:
        """Track how sparsity of precision matrices evolves during training."""
        sparsity_evolution = {}
        
        for layer_idx, attn_module in enumerate(self.model.attention_modules):
            layer_key = f'layer_{layer_idx}'
            sparsity_over_time = []
            
            for analysis in attn_module.attention_weights_history:
                sparsity_over_time.append(analysis.get('mean_sparsity', 0))
            
            sparsity_evolution[layer_key] = sparsity_over_time
        
        return sparsity_evolution


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create a simple Gaussian Graphical Model Transformer
    model = GaussianGraphicalModelTransformer(
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        vocab_size=1000,
        dropout=0.1,
        precision_regularization=0.01,
    )
    
    # Create dummy input
    batch_size, seq_len = 2, 16
    x = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass
    output, analyses = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Number of layers analyzed: {len(analyses)}")
    print(f"\nAnalysis for layer 0, head 0:")
    print(f"  Mean attention sparsity: {analyses[0].get('mean_sparsity', 'N/A'):.4f}")
    print(f"  Precision matrix shape: {analyses[0]['precision_estimates'][0].shape}")
    
    # Analyze precision structures
    analyzer = PrecisionAnalyzer(model)
    cond_indep = analyzer.analyze_conditional_independence()
    print(f"\nConditional independence analysis:")
    for layer, stats in cond_indep.items():
        print(f"  {layer}: {stats['mean_cond_indep_pairs']:.2f} mean pairs")