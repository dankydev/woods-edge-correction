from pathlib import Path
import torch
from multiresunet.correction_dataset import WoodCorrectionDataset
from multiresunet.losses import WoodCorrectionLoss
from multiresunet.model import MultiResUnet
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from utils.progress_bar import ProgressBar
from torch.utils.tensorboard import SummaryWriter

# TODO: Parametrization

model = MultiResUnet(channels=3, filters=16, nclasses=1).to('cuda')
loss = WoodCorrectionLoss(mse_w=10, ssim_w=11, vgg_w=0)
dataset = WoodCorrectionDataset(
        dataset_path=Path("../dataset/Legni02@resize_16x_TRAIN"),
        cut_size_h_w=(128, 256),
        max_shift=15,
        min_shift=0,
        test_mode=False)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=2,
    num_workers=4,
    shuffle=True,
    drop_last=True
)

dataset_iterator = iter(dataloader)

optimizer = Adam(model.parameters(), lr=0.0002)
epochs = 100

summary_writer = SummaryWriter("../logs/")
progress_bar = ProgressBar(max_step=len(dataloader), max_epoch=epochs)

model_name = f"10mse11ssim100epochs.pth"

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

        if global_step % 1000 == 0:
            summary_writer.add_images(tag="inputs", img_tensor=batch[0], global_step=global_step)
            summary_writer.add_images(tag="targets", img_tensor=batch[1], global_step=global_step)
            summary_writer.add_images(tag="outputs", img_tensor=predicted, global_step=global_step)
            summary_writer.add_scalar(tag='train_loss', scalar_value=l, global_step=global_step)

        global_step = global_step + 1

    torch.save(model.state_dict(), f"../models/{model_name}")

