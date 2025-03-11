import wandb
run = wandb.init()
artifact = run.use_artifact('aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/model_checkpoint:v53', type='model')
artifact_dir = artifact.download()