if __name__ == "__main__":
    import os.path
    from pathlib import Path
    import torch
    import torchvision
    import matplotlib.pyplot as plt
    import yaml
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image

    from unet.correction_dataset import WoodCorrectionDataset
    from unet.model.unet import UNet

    with open("../conf/generate_refinement_dataset.yaml", "r") as file_stream:
        try:
            conf = yaml.safe_load(file_stream)
        except yaml.YAMLError:
            print("Unable to open configuration file. Aborting")
            exit()

    dataset_path, cut_h, cut_w = conf.get("originalDatasetPath"), conf.get("cutSizeH"), conf.get("cutSizeW")
    max_shift, min_shift = conf.get("maxShift"), conf.get("minShift")
    final_dataset_total_images = conf.get("finalDatasetTotalImages")
    limit_generated = final_dataset_total_images is not None
    dataset_save_path = conf.get("datasetSavePath")

    dataset = WoodCorrectionDataset(
        dataset_path=Path(dataset_path),
        cut_size_h_w=(cut_h, cut_w),
        max_shift=max_shift,
        min_shift=min_shift,
        test_mode=True)

    test_samples = conf.get("testSamples")
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=4,
        shuffle=True,
        drop_last=True
    )

    replace_maxpool_with_stride = conf.get('replaceMaxpoolWithStride')
    channels = conf.get('channels')
    device = conf.get('device')
    model = UNet(n_channels=channels, replace_maxpool_with_stride=replace_maxpool_with_stride, device=device).to(device)
    models_dir = conf.get("modelsDir")
    model_name = conf.get('testModel')
    try:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name), map_location=device))
    except:
        print("Couldn't load the weight file. Are you sure the model is set like the weight's configuration?")
        exit(-1)

    for i, couple in enumerate(data_loader):
        if limit_generated and i > final_dataset_total_images:
            break

        misaligned = couple[0]
        truth = couple[1].to(device)

        corrected = model(misaligned.to(device))
        to_save = torch.cat((corrected, truth), dim=2)

        save_image(to_save, os.path.join(dataset_save_path, f"{i}.png"))
        print(f"Saved image with id {i}")


