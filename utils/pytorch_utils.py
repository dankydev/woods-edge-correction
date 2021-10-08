from torchvision import transforms


def show_tensor(tensor):
    transforms.ToPILImage(mode="RGB")(tensor.cpu()).show()
