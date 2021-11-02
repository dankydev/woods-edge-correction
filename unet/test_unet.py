if __name__ == "__main__":
    import os.path
    from pathlib import Path

    import torch
    import torchvision
    import matplotlib.pyplot as plt
    import yaml
    from torch.utils.data import DataLoader

    from unet.correction_dataset import WoodCorrectionDataset
    from unet.model.unet import UNet

    with open("../conf/test.yaml", "r") as file_stream:
        try:
            conf = yaml.safe_load(file_stream)
        except yaml.YAMLError:
            print("Unable to open configuration file. Aborting")
            exit()

    dataset_path, cut_h, cut_w = conf.get("testDatasetPath"), conf.get("cutSizeH"), conf.get("cutSizeW")
    max_shift, min_shift = conf.get("maxShift"), conf.get("minShift")

    dataset = WoodCorrectionDataset(
        dataset_path=Path(dataset_path),
        cut_size_h_w=(cut_h, cut_w),
        max_shift=max_shift,
        min_shift=min_shift,
        test_mode=True)

    test_samples = conf.get("testSamples")
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=test_samples,
        num_workers=4,
        shuffle=True,
        drop_last=True
    )

    dataset_iterator = iter(data_loader)
    to_test = next(dataset_iterator)

    replace_maxpool_with_stride = conf.get('replaceMaxpoolWithStride')
    channels = conf.get('channels')
    device = conf.get('device')
    model = UNet(n_channels=channels, replace_maxpool_with_stride=replace_maxpool_with_stride, device=device).to(device)

    to_test = [to_test[0].to(device), to_test[1].to(device)]

    models_dir = conf.get("modelsDir")
    model_name = conf.get('testModel')
    try:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name), map_location=device))
    except:
        print("Couldn't load the weight file. Are you sure the model is set like the weight's configuration?")
        exit(-1)

    misaligned = to_test[0]
    truth = to_test[1]

    results = model(misaligned)

    results_to_show = results[0, :, :, :]
    for i in range(1, results.size()[0]):
        results_to_show = torch.cat((results_to_show, results[i]), dim=1)

    misaligned_to_show = misaligned[0]
    for i in range(1, misaligned.size()[0]):
        misaligned_to_show = torch.cat((misaligned_to_show, misaligned[i]), dim=1)

    truth_to_show = truth[0]
    for i in range(1, truth.size()[0]):
        truth_to_show = torch.cat((truth_to_show, truth[i]), dim=1)

    to_show = torch.cat((misaligned_to_show, results_to_show, truth_to_show), dim=2)

    plt.title("Input/Results/Output")
    plt.imshow(torchvision.transforms.ToPILImage()(to_show))

    plt.waitforbuttonpress()

