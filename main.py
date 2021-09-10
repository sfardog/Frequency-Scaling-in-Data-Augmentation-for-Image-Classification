import warnings
warnings.filterwarnings('ignore')

import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from utils import *
from models import *

def main(args) :
    print('hello world!')
    device = get_deivce()
    fix_seed(device)

    test_transform = transform_generator(args)

    dataset_rootdir = os.path.join('.', args.data_path)

    try:
        dataset_dir = os.path.join(dataset_rootdir, args.data_type)
    except TypeError:
        print("join() argument must be str, bytes, or os.PathLike object, not 'NoneType'")
        print("Please explicitely write the dataset type")
        sys.exit()

    test_data = datasets.CIFAR10(dataset_dir, train=False,
                                 transform=test_transform, download=True)
    num_classes = 10

    test_loader = DataLoader(test_data,
                             batch_size = args.batch_size,
                             shuffle=False,
                             num_workers = args.num_workers,
                             pin_memory=True,
                             worker_init_fn=seed_worker)

    model = Classifier(args.model_name, num_classes).model
    if args.parallel: model = nn.DataParallel(model)
    model.to(device)

    test_result = test(device, args, model, test_loader, eecp=True if args.augment=='eecp' else False)
    save_result(args, test_result, eecp=True if args.augment=='eecp' else False)

def test(device, args, model, test_loader, eecp) :
    model, epochs, model_name = load_model(args, eecp, model_dir='CIFAR/model_save')

    model.eval()
    total_loss, total_correct, correct_top1 = 0., 0, 0
    total = 0

    with torch.no_grad():
        for batch_idx, (image, target) in enumerate(test_loader):
            if (batch_idx + 1) % args.step == 0:
                print("{}/{}({}%) COMPLETE".format(
                    batch_idx + 1, len(test_loader), np.round((batch_idx + 1) / len(test_loader) * 100), 4))

            image, target = image.to(device).float(), target.to(device).long()
            out = model(image)
            loss = F.cross_entropy(out, target)
            total_loss += loss.item()

            # Calculate top 1 accuracy
            _, rank1 = torch.max(out, 1)
            total += target.size(0)

            correct_top1 += (rank1 == target).sum().item()

    test_loss = np.round(total_loss / len(test_loader.dataset), 6)
    test_top1_acc = np.round(correct_top1 / total, 6)

    if (batch_idx + 1) % args.step == 0 or (batch_idx + 1) == len(test_loader):
        print("Epoch {} | test loss : {} | test top1 acc : {}".format(epochs, test_loss, test_top1_acc))

    return test_loss, test_top1_acc

if __name__ == '__main__' :
    args = argparsing()
    main(args)