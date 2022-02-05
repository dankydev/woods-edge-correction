if __name__ == "__main__":
    import os
    import piq
    import torch
    import matplotlib.pyplot as plt
    import torchvision

    from pathlib import Path

    from torch.utils.data import DataLoader

    from unet.correction_dataset import WoodCorrectionDataset
    from unet.model.unet import UNet
    import yaml

    with open("../conf/eval.yaml", "r") as file_stream:
        try:
            conf = yaml.safe_load(file_stream)
        except yaml.YAMLError:
            print("Unable to open configuration file. Aborting")
            exit()

    dataset_path, cut_h, cut_w = conf.get("evalDatasetPath"), conf.get("cutSizeH"), conf.get("cutSizeW")
    max_shift, min_shift = conf.get("maxShift"), conf.get("minShift")
    samples = conf.get("samples")

    dataset = WoodCorrectionDataset(
        dataset_path=Path(dataset_path),
        cut_size_h_w=(cut_h, cut_w),
        max_shift=max_shift,
        min_shift=min_shift,
        test_mode=True)

    if samples is None:
        samples = len(dataset)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=samples,
        num_workers=4,
        shuffle=True,
        drop_last=True
    )

    iterator = iter(data_loader)
    data = next(iterator)

    replace_maxpool_with_stride = conf.get('replaceMaxpoolWithStride')
    channels = conf.get('channels')
    device = conf.get('device')
    model = UNet(n_channels=channels, replace_maxpool_with_stride=replace_maxpool_with_stride, device=device).to(device)

    models_dir = conf.get("modelsDir")
    model_name = conf.get('evalModel')

    model.load_state_dict(torch.load(os.path.join(models_dir, model_name), map_location=device))
    print(model.eval())

    misaligned = data[0].to(device)
    aligned = data[1].to(device)

    predictions = model(misaligned)

    psnr = piq.psnr(predictions, aligned)
    ssim = piq.ssim(predictions, aligned)
    mse = torch.nn.MSELoss()(predictions, aligned)
    brisque = piq.brisque(predictions)
    dss = piq.dss(predictions, aligned)
    # dists = piq.DISTS()(predictions, aligned)

    print(f"PSNR:\t\t{psnr} dB")
    print(f"SSIM:\t\t{ssim}")
    print(f"MSE:\t\t{mse}")
    print(f"BRISQUE:\t{brisque}")
    # print(f"DISTS:\t\t{dists}")

    show_results = conf.get("showResults")

    if show_results:
        results_to_show = predictions[0, :, :, :]
        for i in range(1, predictions.size()[0]):
            results_to_show = torch.cat((results_to_show, predictions[i]), dim=1)

        misaligned_to_show = misaligned[0]
        for i in range(1, misaligned.size()[0]):
            misaligned_to_show = torch.cat((misaligned_to_show, misaligned[i]), dim=1)

        truth_to_show = aligned[0]
        for i in range(1, aligned.size()[0]):
            truth_to_show = torch.cat((truth_to_show, aligned[i]), dim=1)

        to_show = torch.cat((misaligned_to_show, results_to_show, truth_to_show), dim=2)

        plt.title("Input/Results/Output")
        plt.imshow(torchvision.transforms.ToPILImage()(to_show))

        plt.waitforbuttonpress()

