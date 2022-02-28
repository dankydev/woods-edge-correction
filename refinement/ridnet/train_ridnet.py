if __name__ == "__main__":
    from pathlib import Path
    import torch
    import os
    from refinement_dataset import RefinementDataset
    from losses import WoodCorrectionLoss
    from ridnet import RIDNET
    from torch.optim.adam import Adam
    from torch.utils.data import DataLoader, Dataset
    from utils.progress_bar import ProgressBar
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    import yaml

    with open("conf/train.yaml", "r") as file_stream:
        try:
            conf = yaml.safe_load(file_stream)
        except yaml.YAMLError:
            print("Unable to open configuration file. Aborting")
            exit()

    channels = conf.get('channels')
    device = conf.get('device')
    model = RIDNET(n_feats=64).to(device)

    models_dir = conf.get("modelsDir")
    starting_model = conf.get("startingModel")
    if starting_model is not None:
        model.load_state_dict(torch.load(os.path.join(models_dir, starting_model)))

    mse_w, vgg_high_w, ms_ssim_w, vgg_low_w, dists_w, dss_w, fsim_w = conf.get("mseWeight"), conf.get(
        "vggHighWeight"), conf.get("msSsimWeight"), conf.get("vggLowWeight"), conf.get("distsWeight"), conf.get(
        "dssWeight"), conf.get("fsimWeight")
    loss = WoodCorrectionLoss(mse_w=mse_w, vgg_high_w=vgg_high_w, ms_ssim_w=ms_ssim_w, vgg_low_w=vgg_low_w,
                              dists_w=dists_w, dss_w=dss_w, device=device, fsim_w=fsim_w).to(device)

    dataset_path = conf.get("trainDatasetPath")
    dataset = RefinementDataset(dataset_path=Path("C:\\Users\\manic\\Documents\\woods-edge-correction\\dataset\\refinement\\output\\train\\wood"))

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
    current_datetime = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    exp_name = conf.get("expName")
    model_name = f"{current_datetime}_{exp_name}_ridnet_mse{mse_w}_vgglow{vgg_low_w}_msssim{ms_ssim_w}_vgghigh{vgg_high_w}_dists{dists_w}_dss{dss_w}_fsim{fsim_w}_lr{lr}_batch{batch_size}_epochs{epochs}_freq{log_step_frequency}.pth"

    logDir = conf.get("logDir")
    log_name = model_name.replace(".pth", "")
    train_log_dir = os.path.join(logDir, log_name)
    os.mkdir(train_log_dir)
    summary_writer = SummaryWriter(train_log_dir)

    progress_bar = ProgressBar(max_step=len(dataloader), max_epoch=epochs)

    global_step = 0
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            predicted = model(batch[0].to(device))
            predicted = predicted.to(device)

            l, mse_l, vgg_h, ms_ssim, vgg_l, dists_l, dss_l, fsim_l = loss(predicted, batch[1].to(device))
            l.backward()
            optimizer.step()

            progress_bar.inc()
            print(f'\r{progress_bar} '
                  f'│ Loss: {l:.2f} '
                  , end='')

            if global_step % log_step_frequency == 0:
                batch[0] = batch[0].to('cpu')
                batch[1] = batch[1].to('cpu')
                predicted = predicted.to('cpu')

                cat = torch.cat((batch[0], predicted), 2)
                cat = torch.cat((cat, batch[1]), 2)
                summary_writer.add_images(tag="results", img_tensor=cat, global_step=global_step)
                summary_writer.add_scalar(tag='total loss', scalar_value=l, global_step=global_step)

                if mse_w > 0:
                    summary_writer.add_scalar(tag='MSE loss', scalar_value=mse_l, global_step=global_step)
                if vgg_high_w > 0:
                    summary_writer.add_scalar(tag='VGG high loss', scalar_value=vgg_h, global_step=global_step)
                if vgg_low_w > 0:
                    summary_writer.add_scalar(tag='VGG low loss', scalar_value=vgg_l, global_step=global_step)
                if ms_ssim_w > 0:
                    summary_writer.add_scalar(tag='MS-SSIM loss', scalar_value=ms_ssim, global_step=global_step)
                if dists_w > 0:
                    summary_writer.add_scalar(tag='DISTS loss', scalar_value=dists_l, global_step=global_step)
                if dss_w > 0:
                    summary_writer.add_scalar(tag='DSS loss', scalar_value=dss_l, global_step=global_step)
                if fsim_w > 0:
                    summary_writer.add_scalar(tag='FSIM loss', scalar_value=fsim_l, global_step=global_step)

            global_step = global_step + 1

        torch.save(model.state_dict(), os.path.join(models_dir, model_name))
