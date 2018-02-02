from model import MobileNetV2
from utils import parse_args, create_experiment_dirs
import torch.backends.cudnn as cudnn
from cifar10data import CIFAR10Data
from train import Train


def main():
    # Parse the JSON arguments
    try:
        config_args = parse_args()
    except:
        print(
            "Add a config file using \'--config file_name.json\'. "
            "If you added it correctly, make sure that it's in a valid json format.")
        exit(1)

    # Create the experiment directories
    _, config_args.summary_dir, config_args.checkpoint_dir = create_experiment_dirs(config_args.experiment_dir)

    model = MobileNetV2(config_args)

    if config_args.cuda:
        model.cuda()
        cudnn.enabled = True
        cudnn.benchmark = True

    print("Loading Data...")
    data = CIFAR10Data(config_args)
    # data.plot_random_sample()
    print("Data loaded successfully\n")

    trainer = Train(model, data.trainloader, data.testloader, config_args)

    if config_args.to_train:
        try:
            print("Training...")
            trainer.train()
            print("Training Finished\n")
        except KeyboardInterrupt:
            pass

    if config_args.to_test:
        print("Testing...")
        trainer.test(data.testloader)
        print("Testing Finished\n")


if __name__ == "__main__":
    main()
