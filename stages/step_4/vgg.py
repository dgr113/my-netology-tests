# coding: utf-8

import zipfile
import matplotlib.pyplot as plt  # type: ignore

from io import BytesIO
from itertools import repeat
from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass, field, InitVar
from urllib.request import urlopen

from torch import Tensor, no_grad, max as torch_max, load as load_model
from torch.nn import Module, Linear
from torch.nn.functional import cross_entropy
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.datasets import ImageFolder, FashionMNIST  # type: ignore
from torchvision.transforms import ToTensor, Compose, Resize, RandomHorizontalFlip, RandomResizedCrop, Normalize  # type: ignore
from torchvision.models import vgg16 as vgg  # type: ignore




@dataclass
class DataLoaderParams:
    batch_size: int = 30
    batch_shuffle: bool = False



@dataclass
class TrainParams:
    epochs: int
    lr: float
    accuracy_threshold: float = 0.9



@dataclass
class TrainStats:
    epoch: int
    loss: float
    accuracy: float
    train_params: 'TrainParams'



class CustomVGG(Module):
    def __init__(self, pretrained: bool = False, out_classes: Optional['int'] = None, classifier: Optional['Module'] = None):
        super().__init__()
        model = vgg(pretrained=pretrained, progress=True)

        if pretrained:
            for layer_param in model.parameters():
                layer_param.requires_grad = False
        if classifier:
            model.classifier = classifier
        if out_classes:
            model.classifier[-1] = Linear(in_features=model.classifier[-1].in_features, out_features=out_classes, bias=True)

        self.ol = model

    def _forward_unimplemented(self, *inp: 'Tensor') -> None:
        pass

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
            for X_batch, y_batch in self._data:
                X_predicted = self.model(X_batch)

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
    def test_model(dataset: 'Dataset', loader_params: 'DataLoaderParams', model: Union[str, 'Path', 'Module']) -> None:
        if isinstance(model, (str, Path)):
            model = CustomVGG()
            model.load_state_dict( load_model(model) )

        total, correct = 0, 0
        with no_grad():
            for images, labels in DataLoader(dataset, batch_size=loader_params.batch_size, shuffle=loader_params.batch_shuffle):
                outputs = model(images)
                _, predicted = torch_max(outputs.data, 1)
                total += labels.size(0)
                correct += int( outputs.argmax(dim=1).eq(labels).sum().item() )

        result_accuracy = 100 * correct / total
        print( "Accuracy of the network on test data is: {}%".format(result_accuracy) )



class DatasetUtils:
    @staticmethod
    def _check_or_load_dataset(dataset_url: str, save_root: Union['Path', str]) -> None:
        if not Path(save_root).exists():
            response = urlopen(dataset_url)
            with zipfile.ZipFile( BytesIO( response.read() ) ) as zf:
                zf.extractall(save_root)

    @staticmethod
    def _visualize_dataset(dataset: 'Dataset') -> None:
        def show_image(inp):
            inp = inp.numpy().transpose((1, 2, 0))
            plt.imshow(inp)
            plt.pause(0.001)

        images_so_far = 0
        plt.figure()

        with no_grad():
            for i, (X, y) in enumerate( DataLoader(dataset, batch_size=256, shuffle=False) ):
                for j in range( X.size()[0] ):
                    images_so_far += 1
                    show_image( X.cpu().data[j] )

    @staticmethod
    def get_mnist_dataset(save_dir: str, transform: Optional['Compose'] = None, is_train: bool = True) -> 'Dataset':
        return FashionMNIST(
            root=save_dir,
            train=is_train,
            download=True,
            transform=transform
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
    def multiply_dataset(dataset_full_path: Union['Path', str], transforms: 'Compose', n_count: int = 3) -> 'Dataset':
        return ConcatDataset(list(repeat(
            ImageFolder(
                Path(dataset_full_path),
                transforms
            ),
            n_count
        )))




def main():
    dataset = DatasetUtils.get_external_dataset(
        'https://download.pytorch.org/tutorial/hymenoptera_data.zip',
        './datasets',
        'hymenoptera_data/train',
        Compose([
            RandomHorizontalFlip(),
            Resize([128, 128]),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    aug_dataset = DatasetUtils.multiply_dataset(
        './datasets/hymenoptera_data/train',
        Compose([
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            Resize([128, 128]),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        n_count=3
    )

    train_params = TrainParams(epochs=10, lr=0.005, accuracy_threshold=0.9)
    train_loader_params = DataLoaderParams(batch_size=256, batch_shuffle=True)

    model = CustomVGG(pretrained=True, out_classes=2)
    start_optimizer = Adam(model.parameters(), lr=train_params.lr, amsgrad=True)

    Processing.train_model(ConcatDataset([dataset, aug_dataset]), train_params, train_loader_params, model, start_optimizer)





if __name__ == '__main__':
    main()
