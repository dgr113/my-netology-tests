# coding: utf-8

import zipfile
from io import BytesIO
from pathlib import Path
from typing import Union, Optional, Mapping
from dataclasses import dataclass, field, InitVar
from urllib.request import urlopen

from torch import Tensor, no_grad, max as torch_max, load as load_model
from torch.nn import Module
from torch.nn.functional import cross_entropy
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder  # type: ignore
from torchvision.transforms import ToTensor, Compose, Normalize, CenterCrop, Resize, RandomHorizontalFlip, RandomResizedCrop  # type: ignore
from torchvision.models import resnet18  # type: ignore




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



class Network(Module):
    def __init__(self):
        super().__init__()
        self.ol = resnet18()

    def forward(self, out: 'Tensor') -> 'Tensor':
        out = self.ol(out)
        return out



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



class DatasetUtils:
    @staticmethod
    def _check_or_load_dataset(dataset_url: str, save_root: Union['Path', str]) -> None:
        if not Path(save_root).exists():
            response = urlopen(dataset_url)
            with zipfile.ZipFile( BytesIO( response.read() ) ) as zf:
                zf.extractall(save_root)


    @staticmethod
    def get_external_dataset(
        dataset_url: str,
        save_root: Union['Path', str],
        dataset_inner_path: Union['Path', str],
        transforms: Optional[Mapping[str, 'Compose']] = None

    ) -> Mapping[str, 'ImageFolder']:

        """ Load external (zipped) dataset from Web

            :param dataset_url: External dataset URL
            :param save_root: Local save directory
            :param dataset_inner_path: Inner path to dataset into archive
            :param transforms: Images transformations pipeline
        """
        DatasetUtils._check_or_load_dataset(dataset_url, save_root)
        return {
            x: ImageFolder(Path(save_root).joinpath(dataset_inner_path).joinpath(x), transforms and transforms[x])
            for x in ('train', 'val')
        }



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
            model = Network()
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




def main():
    data_transforms = {
        'train': Compose([
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'val': Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }
    # data_transforms = {
    #     'train': Compose([
    #         ToTensor(),
    #         Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    #     'val': Compose([
    #         ToTensor(),
    #         Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    # }

    dataset = DatasetUtils.get_external_dataset('https://download.pytorch.org/tutorial/hymenoptera_data.zip', './datasets', 'hymenoptera_data', data_transforms)
    # test_dataset = Processing.get_mnist_dataset('./datasets/FashionMNIST_TEST', is_train=False)

    train_params = TrainParams(epochs=10, lr=0.005, accuracy_threshold=0.9)
    train_loader_params = DataLoaderParams(batch_size=256, batch_shuffle=True)
    # test_loader_params = DataLoaderParams(batch_size=len(test_dataset), batch_shuffle=False)

    model = Network()
    start_optimizer = SGD(model.parameters(), lr=train_params.lr, momentum=0.9)

    Processing.train_model(dataset['train'], train_params, train_loader_params, model, start_optimizer)




if __name__ == '__main__':
    main()
