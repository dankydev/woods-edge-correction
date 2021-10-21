from pathlib import Path
import torch
import os
from correction_dataset import WoodCorrectionDataset
from losses import WoodCorrectionLoss
from model.unet import UNet
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from utils.progress_bar import ProgressBar
from torch.utils.tensorboard import SummaryWriter
import yaml

with open("../conf/default.yaml", "r") as file_stream:
    try:
        conf = yaml.safe_load(file_stream)
    except yaml.YAMLError:
        print("Unable to open configuration file. Aborting")
        exit()

replace_maxpool_with_stride = conf.get('replaceMaxpoolWithStride')
channels = conf.get('channels')
model = UNet(n_channels=channels, replace_maxpool_with_stride=replace_maxpool_with_stride).to('cuda')

models_dir = conf.get("modelsDir")
starting_model = conf.get("startingModel")
if starting_model is not None:
    model.load_state_dict(torch.load(os.path.join(models_dir, starting_model)))

mse_w, ms_ssim_w, vgg_w = conf.get("mseWeight"), conf.get("msSsimWeight"), conf.get("vggWeight")
loss = WoodCorrectionLoss(mse_w=mse_w, ms_ssim_w=ms_ssim_w, vgg_w=vgg_w)

dataset_path, cut_h, cut_w = conf.get("trainDatasetPath"), conf.get("cutSizeH"), conf.get("cutSizeW")
max_shift, min_shift = conf.get("maxShift"), conf.get("minShift")
dataset = WoodCorrectionDataset(
    dataset_path=Path(dataset_path),
    cut_size_h_w=(cut_h, cut_w),
    max_shift=max_shift,
    min_shift=min_shift,
    test_mode=False)

batch_size, num_workers, shuffle = conf.get("batchSize"), conf.get("numWorkers"), conf.get("shuffle")
dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=shuffle,
    drop_last=True
)

dataset_iterator = iter(dataloader)

lr, epochs = conf.get("lr"), conf.get("epochs")
optimizer = Adam(model.parameters(), lr=lr)

log_step_frequency = conf.get("logStepFrequency") 
model_name = f"unet_maxshift{max_shift}_minshift{min_shift}_mse{mse_w}_ssim{ms_ssim_w}_vgg{vgg_w}_lr{lr}_batch{batch_size}_epochs{epochs}_freq{log_step_frequency}_replaceWithStride{replace_maxpool_with_stride}.pth"

logDir = conf.get("logDir")
log_name = model_name.replace(".pth", "")
train_log_dir = os.path.join(logDir, log_name)
if os.path.exists(train_log_dir):
    os.system(f"rm -rf {train_log_dir}")
os.mkdir(train_log_dir)
summary_writer = SummaryWriter(train_log_dir)

progress_bar = ProgressBar(max_step=len(dataloader), max_epoch=epochs)

global_step = 0
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        predicted = model(batch[0].to('cuda'))

        l = loss(predicted, batch[1].to('cuda'))
        l.backward()
        optimizer.step()

        progress_bar.inc()
        print(f'\r{progress_bar} '
              f'│ Loss: {l:.15f} '
              , end='')

        if global_step % log_step_frequency == 0:            
            batch[0] = batch[0].to('cpu')
            batch[1] = batch[1].to('cpu')
            predicted = predicted.to('cpu')
            
            cat = torch.cat((batch[0], predicted), 2)
            cat = torch.cat((cat, batch[1]), 2)
            summary_writer.add_images(tag="results", img_tensor=cat, global_step=global_step)
            summary_writer.add_scalar(tag='train_loss', scalar_value=l, global_step=global_step)

        global_step = global_step + 1

    torch.save(model.state_dict(), os.path.join(models_dir, model_name))
