"""Debug from_config."""
from papers.radon_cnn.models.radon_cnn import RadonCNN

d = {"in_channels": 3, "num_classes": 5, "radon_theta": 32}
m = RadonCNN.from_config(d)
print(f"theta: {m.radon.theta}")
print(f"fc3.out_features: {m.classifier.fc3.out_features}")
print(f"conv1.in_channels: {m.conv1.conv.in_channels}")