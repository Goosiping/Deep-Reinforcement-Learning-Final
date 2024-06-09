import pandas as pd
import wandb

wandb.init(project="Montezuma's Revenge")

df = pd.read_csv('./embeddings.csv')
wandb.log({'embeddings': df})
wandb.finish()