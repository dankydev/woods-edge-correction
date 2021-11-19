if __name__ == "__main__":
    import os
    import torch
    import matplotlib.pyplot as plt
    from PIL import Image
    import torchvision.transforms.functional as TF
    import yaml
    from unet.model.unet import UNet


    def show_tensor_image(image: torch.Tensor):
        plt.figure()
        plt.imshow(image.detach().cpu().permute(1, 2, 0))
        plt.waitforbuttonpress()


    def get_portion(image, start, end, portion_size=128):
        assert start < end

        return image[:, :, start*portion_size:end*portion_size]


    def correct_multiple_passes(image: torch.Tensor, verbose=True, portion_size=128):
        width_portions = image.size()[-1] // portion_size
        if verbose:
            show_tensor_image(image)
        result_image = correct_image(image, verbose=verbose)
        result_image = correct_image_double_pass(result_image, verbose=verbose)

        if verbose:
            show_tensor_image(result_image)

        for p in range(1, width_portions):
            for rp in range(p):
                to_correct = get_portion(result_image, rp, rp + 2)
                corrected = model(torch.unsqueeze(to_correct, 0))[0]
                result_image[:, :, rp*portion_size:(rp+2)*portion_size] = corrected

        if verbose:
            show_tensor_image(result_image)

        return result_image


    # TODO: generalize range limits for even and odd numbers
    def correct_image_double_pass(image: torch.Tensor, verbose=True, portion_size=128):
        width_portions = image.size()[-1] // portion_size
        if verbose:
            show_tensor_image(image)
        result_image = None

        # first pass (even pos)
        for p in range(0, width_portions, 2):
            portion_to_correct = image[:, :, p * portion_size:(p + 2) * portion_size]
            portion_corrected = model(torch.unsqueeze(portion_to_correct, 0))[0]
            if result_image is None:
                result_image = portion_corrected
            else:
                result_image = torch.cat((result_image, portion_corrected), dim=2)

        if verbose:
            show_tensor_image(result_image)

        # second pass (odd pos)
        for p in range(1, width_portions - 1, 2):
            portion_to_correct = result_image[:, :, p * portion_size:(p + 2) * portion_size]
            portion_corrected = model(torch.unsqueeze(portion_to_correct, 0))[0]
            result_image[:, :, (p + 1) * portion_size:(p + 2) * portion_size] = portion_corrected[:, :, portion_size:portion_size*2]

        if verbose:
            show_tensor_image(result_image)

        return result_image


    def correct_image(image: torch.Tensor, verbose=True, portion_size=128):
        width_portions = image.size()[-1] // portion_size
        if verbose:
            show_tensor_image(image)
        result_image = None

        for p in range(width_portions):
            if result_image is None:
                portion_to_correct = get_portion(image, p, p+2, portion_size=portion_size)
                portion_corrected = model(torch.unsqueeze(portion_to_correct, 0))[0]
                result_image = portion_corrected
            else:
                # left_half_to_correct = result_image[:, :, -128:]
                left_half_to_correct = get_portion(image, p, p+1, portion_size=portion_size)
                right_half_to_correct = get_portion(image, p+1, p+2, portion_size=portion_size)
                to_correct = torch.cat((left_half_to_correct, right_half_to_correct), dim=2)
                corrected = model(torch.unsqueeze(to_correct, 0))[0]
                result_image = torch.cat((result_image, corrected[:, :, portion_size:portion_size*2]), dim=2)

        if verbose:
            show_tensor_image(result_image)
        return result_image


    def correct_images(path, portion_size=128):
        images = os.listdir(path)
        images = [x for x in images if os.path.isfile(os.path.join(path, x))]
        for im in images:
            current_image = Image.open(os.path.join(path, im))
            current_image = TF.to_tensor(current_image)
            current_image = current_image.to('cpu')

            show_tensor_image(current_image)
            corrected_image = correct_image(current_image, verbose=False, portion_size=portion_size)
            show_tensor_image(corrected_image)


    with open("../conf/correct.yaml", "r") as file_stream:
        try:
            conf = yaml.safe_load(file_stream)
        except yaml.YAMLError:
            print("Unable to open configuration file. Aborting")
            exit()

    replace_maxpool_with_stride = conf.get('replaceMaxpoolWithStride')
    channels = conf.get('channels')
    device = conf.get('device')
    model = UNet(n_channels=channels, replace_maxpool_with_stride=replace_maxpool_with_stride, device=device).to(device)
    correction_model = conf.get('correctModel')
    models_dir = conf.get('modelsDir')

    model.load_state_dict(torch.load(os.path.join(models_dir, correction_model)))

    path_to_correct = conf.get('pathToCorrect')
    portionsize = conf.get('portionSize')

    if os.path.isdir(path_to_correct):
        correct_images(path_to_correct, portion_size=portionsize)
    elif os.path.isfile(path_to_correct):
        pass
    else:
        print(f"Error: {path_to_correct} is not a file or a directory")
