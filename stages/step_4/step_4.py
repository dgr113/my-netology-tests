# coding: utf-8

import zipfile
import matplotlib.pyplot as plt  # type: ignore

from io import BytesIO
from pathlib import Path
from itertools import repeat, starmap
from typing import Union, Optional, Tuple
from dataclasses import dataclass, field, InitVar
from urllib.request import urlopen

from torch import Tensor, no_grad, device, cuda,  max as torch_max, load as load_model  # type: ignore
from torch.nn import Module, Linear  # type: ignore
from torch.nn.functional import cross_entropy  # type: ignore
from torch.optim import Optimizer, Adam, SGD  # type: ignore
from torch.utils.data import DataLoader, Dataset, ConcatDataset  # type: ignore
from torchvision.datasets import ImageFolder, FashionMNIST  # type: ignore
from torchvision.transforms import ToTensor, Compose, Resize, RandomHorizontalFlip, RandomResizedCrop, Normalize, Grayscale, RandomAffine, RandomGrayscale, RandomRotation, ColorJitter  # type: ignore
from torchvision.transforms import RandomVerticalFlip
from torchvision.models import vgg16 as vgg, resnet18 as resnet  # type: ignore

TORCH_DEVICE = device( 'cuda' if cuda.is_available() else 'cpu' )  # USE CUDA GPU




@dataclass
class DataLoaderParams:
    batch_size: int = 30
    batch_shuffle: bool = False



@dataclass
class TrainParams:
    epochs: int
    lr: float
    accuracy_threshold: float = 0.85



@dataclass
class TrainStats:
    epoch: int
    loss: float
    accuracy: float
    train_params: 'TrainParams'



class CustomVGG(Module):
    def __init__(self, *, pretrained: bool = False, out_classes: Optional['int'] = None):
        super().__init__()
        model = vgg(pretrained=pretrained, progress=False)

        if pretrained:
            for layer_param in model.parameters():
                layer_param.requires_grad = False

        if out_classes:
            model.classifier[-1] = Linear(in_features=model.classifier[-1].in_features, out_features=out_classes, bias=True)

        self.ol = model

    def forward(self, t: 'Tensor') -> 'Tensor':
        return self.ol(t)



class CustomResNet(Module):
    def __init__(self, *, pretrained: bool = False, out_classes: Optional['int'] = None):
        super().__init__()
        model = resnet(pretrained=pretrained, progress=False)

        if pretrained:
            for layer_param in model.parameters():
                layer_param.requires_grad = False

        if out_classes:
            model.fc = Linear(in_features=model.fc.in_features, out_features=out_classes, bias=True)

        self.ol = model

    def forward(self, t: 'Tensor') -> 'Tensor':
        return self.ol(t)



@dataclass
class TrainContext:
    dataset: InitVar['Dataset']

    train_params: 'TrainParams'
    loader_params: 'DataLoaderParams'

    model: 'Module'
    start_optimizer: 'Optimizer'
    final_optimizer: Optional['Optimizer'] = None

    _current_epoch: int = 0
    _epoch_loss: float = 0
    _current_accuracy: float = 0
    _correct_pred_count: int = 0
    _data: 'DataLoader' = field(init=False)

    def __post_init__(self, dataset: 'Dataset'):
        self._data = DataLoader(dataset, batch_size=self.loader_params.batch_size, shuffle=self.loader_params.batch_shuffle)


    def __iter__(self):
        return self


    def __next__(self):
        """ Next train epoch """

        current_optimizer = self.final_optimizer \
            if ( self.final_optimizer and self._current_accuracy > self.train_params.accuracy_threshold ) \
            else self.start_optimizer

        if self._current_epoch < self.train_params.epochs:
            self._epoch_start()
            for X_batch, y_batch in DatasetUtils.loader_to_device(self._data):
                X_predicted = self.model( X_batch )

                loss = cross_entropy(X_predicted, y_batch)
                current_optimizer.zero_grad()

                loss.backward()
                current_optimizer.step()

                self._epoch_update(loss, X_predicted, y_batch)

            epoch_stats = self._get_epoch_stats()
            self._current_accuracy = epoch_stats.accuracy
            return epoch_stats
        else:
            raise StopIteration


    def _epoch_start(self) -> None:
        self._current_epoch += 1
        self._epoch_loss = 0
        self._correct_pred_count = 0


    def _epoch_update(self, loss: 'Tensor', y_pred: 'Tensor', y_target: 'Tensor') -> None:
        """ Update inner epoch stats

            :param loss: current training loss
            :param y_pred: Predicted labels - 2D Array with probabilities of matching each class for each row
            :param y_target: Array of target labels for each row of current data batch
        """
        self._epoch_loss += loss.item() * self._data.batch_size
        self._correct_pred_count += int( y_pred.argmax(dim=1).eq(y_target).sum().item() )


    def _get_epoch_stats(self) -> 'TrainStats':
        """ Calculate main epoch stats """
        dataset_len = len(self._data.dataset)
        return TrainStats(
            epoch=self._current_epoch,
            loss=self._epoch_loss / dataset_len,
            accuracy=self._correct_pred_count / dataset_len,
            train_params=self.train_params
        )



class VizUtils:
    @staticmethod
    def show_image(inp: 'Tensor') -> None:
        inp = inp.numpy().transpose((1, 2, 0))
        plt.imshow(inp)
        plt.pause(0.001)

    @staticmethod
    def visualize_dataset(dataset: 'Dataset') -> None:
        plt.figure()
        with no_grad():
            for i, (X, y) in enumerate( DataLoader(dataset, batch_size=256, shuffle=False) ):
                for j in range( X.size()[0] ):
                    VizUtils.show_image( X.cpu().data[j] )



class DatasetUtils:
    @staticmethod
    def _check_or_load_dataset(dataset_url: str, save_root: Union['Path', str]) -> None:
        if not Path(save_root).exists() or not list( Path(save_root).iterdir() ):
            response = urlopen(dataset_url)
            with zipfile.ZipFile( BytesIO( response.read() ) ) as zf:
                zf.extractall(save_root)

    @staticmethod
    def _batch_to_device(X: 'Tensor', y: 'Tensor') -> Tuple['Tensor', 'Tensor']:
        return X.to(TORCH_DEVICE), y.to(TORCH_DEVICE)


    @staticmethod
    def loader_to_device(data: 'DataLoader') -> Tuple['Tensor', 'Tensor']:
        yield from starmap(DatasetUtils._batch_to_device, data)


    @staticmethod
    def get_mnist_dataset(save_dir: str, is_train: bool = True, transform: Optional['Compose'] = None) -> 'Dataset':
        return FashionMNIST(
            root=save_dir,
            train=is_train,
            download=True,
            transform=transform if transform else Compose([
                ToTensor()
            ])
        )

    @staticmethod
    def get_external_dataset(
            dataset_url: str,
            save_root_path: Union['Path', str],
            dataset_inner_path: Union['Path', str],
            transform: Optional['Compose'] = None

    ) -> 'Dataset':

        """ Load external (zipped) dataset from Web

            :param dataset_url: External dataset URL
            :param save_root_path: Local save directory
            :param dataset_inner_path: Inner path to dataset into archive
            :param transform: Images transformations pipeline
        """
        DatasetUtils._check_or_load_dataset(dataset_url, save_root_path)
        return ImageFolder(
            Path(save_root_path).joinpath(dataset_inner_path),
            transform
        )

    @staticmethod
    def multiply_dataset(dataset_full_path: Union['Path', str], transforms: 'Compose', n: int = 5) -> 'Dataset':
        return ConcatDataset(list(repeat(
            ImageFolder(
                Path(dataset_full_path),
                transforms
            ),
            n
        )))



class Processing:
    @staticmethod
    def train_model(
            dataset: 'Dataset',
            train_params: 'TrainParams',
            loader_params: 'DataLoaderParams',
            model: 'Module',
            start_optimizer: 'Optimizer',
            final_optimizer: Optional['Optimizer'] = None

    ) -> None:

        for epoch_stats in TrainContext(dataset, train_params, loader_params, model, start_optimizer, final_optimizer):
            print( "Epoch train stats: {}".format(epoch_stats) )


    @staticmethod
    def test_model(dataset: 'Dataset', loader_params: 'DataLoaderParams', model: Union[str, 'Path', 'Module']) -> float:
        if isinstance(model, (str, Path)):
            model = CustomVGG()
            model.load_state_dict( load_model(model) )

        total, correct = 0, 0
        with no_grad():
            for images, labels in DatasetUtils.loader_to_device( DataLoader(dataset, batch_size=loader_params.batch_size, shuffle=loader_params.batch_shuffle) ):
                outputs = model(images)
                _, predicted = torch_max(outputs.data, 1)
                total += labels.size(0)
                correct += int( outputs.argmax(dim=1).eq(labels).sum().item() )

        result_accuracy = 100 * correct / total
        print( "Accuracy of the network on test data is: {}%".format(result_accuracy) )
        return result_accuracy



@dataclass
class ModelsTests:
    train_params: 'TrainParams' = TrainParams(epochs=10, lr=0.002)

    def _test_hymenoptera(self, model: 'Module', with_augm: bool = False):
        train_dataset = DatasetUtils.get_external_dataset(
            'https://download.pytorch.org/tutorial/hymenoptera_data.zip',
            './datasets',
            'hymenoptera_data/train',
            Compose([
                Resize([128, 128]),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        )
        if with_augm:
            train_dataset = ConcatDataset([
                DatasetUtils.multiply_dataset(
                    './datasets/hymenoptera_data/train',
                    Compose([
                        RandomResizedCrop(224, scale=(0.5, 1.0)),
                        RandomHorizontalFlip(p=0.5),
                        RandomVerticalFlip(p=0.5),
                        RandomGrayscale(p=0.2),
                        RandomRotation(degrees=0.5),
                        Resize([128, 128]),
                        ToTensor(),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                    19
                ),
                train_dataset
            ])

        test_dataset = DatasetUtils.get_external_dataset(
            'https://download.pytorch.org/tutorial/hymenoptera_data.zip',
            './datasets',
            'hymenoptera_data/val',
            Compose([
                Resize([128, 128]),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        )
        train_loader_params = DataLoaderParams(batch_size=256, batch_shuffle=True)
        test_loader_params = DataLoaderParams(batch_size=256, batch_shuffle=False)

        start_optimizer = Adam(model.parameters(), lr=0.005)
        final_optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)

        Processing.train_model(train_dataset, self.train_params, train_loader_params, model, start_optimizer, final_optimizer)
        Processing.test_model(test_dataset, test_loader_params, model)


    def _test_fmnist(self, model: 'Module'):
        train_dataset = DatasetUtils.get_mnist_dataset(
            './datasets/FMNIST_TRAIN',
            is_train=True,
            transform=Compose([
                Grayscale(3),
                Resize((128, 128)),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        )
        test_dataset = DatasetUtils.get_mnist_dataset(
            './datasets/FMNIST_TEST',
            is_train=False,
            transform=Compose([
                Grayscale(3),
                Resize((128, 128)),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        )
        train_loader_params = DataLoaderParams(batch_size=256, batch_shuffle=True)
        test_loader_params = DataLoaderParams(batch_size=256, batch_shuffle=False)

        start_optimizer = Adam(model.parameters(), lr=0.005)
        final_optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)

        Processing.train_model(train_dataset, self.train_params, train_loader_params, model, start_optimizer, final_optimizer)
        Processing.test_model(test_dataset, test_loader_params, model)


    def vgg_tests_pack(self, pretrained: bool):
        m1 = CustomVGG(pretrained=pretrained, out_classes=2).to(TORCH_DEVICE)

        self._test_hymenoptera(m1, with_augm=False)
        self._test_hymenoptera(m1, with_augm=True)


    def resnet_tests_pack(self, pretrained: bool):
        m1 = CustomVGG(pretrained=pretrained, out_classes=2).to(TORCH_DEVICE)
        m2 = CustomVGG(pretrained=pretrained, out_classes=10).to(TORCH_DEVICE)

        self._test_hymenoptera(m1, with_augm=False)
        self._test_hymenoptera(m1, with_augm=True)
        self._test_fmnist(m2)





def main():
    tests_context = ModelsTests()

    tests_context.vgg_tests_pack(pretrained=True)
    # tests_context.resnet_tests_pack(pretrained=True)

    # tests_context.vgg_tests(pretrained=False)
    # tests_context.resnet_tests(pretrained=False)




if __name__ == '__main__':
    main()
