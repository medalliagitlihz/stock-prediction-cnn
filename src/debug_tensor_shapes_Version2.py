import numpy as np
import torch
import torch.nn as nn

# Create sample data like your code does
SEQ_LENGTH = 50
BATCH_SIZE = 4

# ❌ WRONG: (batch, seq_length, 1)
wrong_shape = np.random.rand(BATCH_SIZE, SEQ_LENGTH, 1)
print(f"❌ Wrong shape: {wrong_shape.shape} → (batch, seq_length, channels)")

# ✅ CORRECT: (batch, 1, seq_length)
correct_shape = np.random.rand(BATCH_SIZE, 1, SEQ_LENGTH)
print(f"✅ Correct shape: {correct_shape.shape} → (batch, channels, seq_length)")

# Test with Conv1d layer
conv = nn.Conv1d(1, 32, kernel_size=3, padding=1)

try:
    output = conv(torch.FloatTensor(wrong_shape))
    print("❌ FAILED: Wrong shape should error!")
except RuntimeError as e:
    print(f"✓ Expected error with wrong shape: {str(e)[:80]}...")

try:
    output = conv(torch.FloatTensor(correct_shape))
    print(f"✓ SUCCESS: Correct shape works! Output shape: {output.shape}")
except RuntimeError as e:
    print(f"❌ FAILED with correct shape: {e}")