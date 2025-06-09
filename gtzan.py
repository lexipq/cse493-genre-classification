import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from model import MLP
from sklearn.manifold import TSNE


mlp = MLP(hidden_dims=[2048, 1024, 512, 128])
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
mlp.load_state_dict(torch.load(f'models/{type(mlp).__name__.lower()}.pth'))
x = torch.load('data/mlp_data.pt')
# y = torch.load('data/mlp_gt.pt')

df = pd.read_csv('data/features_3_sec.csv')
df = df.drop(labels='filename', axis=1)
classes = df.iloc[:,-1]

mlp.to(device)
mlp.eval()

# before transform
tsne = TSNE(n_components=2)
tsne_out = tsne.fit_transform(x.numpy())

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('t-SNE of Hidden Layer Output')
sns.scatterplot(x=tsne_out[:,0], y=tsne_out[:,1], hue=classes, palette='muted', ax=axes[0])
x = x.to(device)

# Get output from second-to-last layer to visualize
with torch.no_grad():
    for layer in mlp.model[:-1]:
        x = layer(x)

tsne = TSNE(n_components=2)
tsne_out = tsne.fit_transform(torch.Tensor.cpu(x).numpy())

sns.scatterplot(x=tsne_out[:,0], y=tsne_out[:,1], hue=classes, palette='muted', ax=axes[1])
plt.tight_layout()
plt.show()

fig.savefig('tsne_before_after.png', dpi=300)
