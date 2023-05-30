import matplotlib as matplotlib
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('batch-16.csv')
df2 = pd.read_csv('batch-32.csv')
df3 = pd.read_csv('batch-64.csv')

ax= df1.plot(x='epoch', y='training loss', style='--', color='b')
df1.plot(x='epoch', y='validation loss', color='b',ax=ax)

df2.plot(x='epoch', y='training loss', style='--', color='r',ax=ax)
df2.plot(x='epoch', y='validation loss', color='r',ax=ax)

df3.plot(x='epoch', y='training loss', style='--', color='g',ax=ax)
df3.plot(x='epoch', y='validation loss', color='g',ax=ax)

plt.legend(['batch_sz = 16 training','batch_sz = 16 validation', 'batch_sz = 32 training','batch_sz = 32 validation','batch_sz = 64 training','batch_sz = 64 validation',])
plt.xlim(0,5)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('learning_curve-batch size')
plt.title('Learning curves for varying batch size - RoBERTa-ViT linear fusion')
plt.show()


df1 = pd.read_csv('hidden - 256.csv')
df2 = pd.read_csv('hidden - 512.csv')
df3 = pd.read_csv('hidden -750.csv')

ax= df1.plot(x='epoch', y='training loss', style='--', color='b')
df1.plot(x='epoch', y='validation loss', color='b',ax=ax)

df2.plot(x='epoch', y='training loss', style='--', color='r',ax=ax)
df2.plot(x='epoch', y='validation loss', color='r',ax=ax)

df3.plot(x='epoch', y='training loss', style='--', color='g',ax=ax)
df3.plot(x='epoch', y='validation loss', color='g',ax=ax)

plt.legend(['hidden_dim = 256 training','hidden_dim = 256 validation', 'hidden_dim = 512 training','hidden_dim = 512 validation','hidden_dim = 750 training','hidden_dim = 750 validation',])
plt.xlim(0,5)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('learning_curve-hidden size')
plt.title('Learning curves for varying hidden dimension of fusion layer')
plt.show()
