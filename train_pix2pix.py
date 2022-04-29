def main():
    import torchvision
    dataset = torchvision.datasets.Cityscapes(
        root="data",
    )


if __name__ == '__main__':
    main()
