# coding: utf-8

from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass, field, InitVar

from torch import Tensor, no_grad, max as torch_max, load as load_model
from torch.nn import Linear, Conv2d, Module
from torch.nn.functional import relu, max_pool2d, cross_entropy
from torch.optim import Adam, Optimizer, SGD
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, MNIST  # type: ignore
from torchvision.transforms import ToTensor, Compose  # type: ignore




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
        self.cl_1 = Conv2d(in_channels=1, out_channels=12, kernel_size=3)
        self.cl_2 = Conv2d(in_channels=12, out_channels=24, kernel_size=3)
        self.ff_1 = Linear(in_features=24 * 5 * 5, out_features=120)
        self.ff_2 = Linear(in_features=120, out_features=60)
        self.out = Linear(in_features=60, out_features=10)

    def forward(self, out: 'Tensor') -> 'Tensor':
        out = max_pool2d(
            relu(self.cl_1(out)),
            kernel_size=2,
            stride=2
        )

        out = max_pool2d(
            relu(self.cl_2(out)),
            kernel_size=2,
            stride=2
        )

        out = relu(self.ff_1(out.reshape(-1, 24 * 5 * 5)))
        out = relu(self.ff_2(out))
        out = self.out(out)
        return out



@dataclass
class TrainContext:
    dataset: InitVar['MNIST']

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


    def __post_init__(self, dataset: 'MNIST'):
        self._data = DataLoader(
            dataset,
            batch_size=self.loader_params.batch_size,
            shuffle=self.loader_params.batch_shuffle
        )


    def __iter__(self):
        return self


    def __next__(self):
        """ Next train epoch """

        # Change optimizer when we get over accuracy threshold
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
    def get_mnist_dataset(save_dir: str, is_train: bool = True) -> 'MNIST':
        return FashionMNIST(
            root=save_dir,
            train=is_train,
            download=True,
            transform=Compose([
                ToTensor()
            ])
        )


    @staticmethod
    def train_model(
        dataset: 'MNIST',
        train_params: 'TrainParams',
        loader_params: 'DataLoaderParams',
        model: 'Module',
        start_optimizer: 'Optimizer',
        final_optimizer: Optional['Optimizer'] = None

    ) -> None:

        for epoch_stats in TrainContext(dataset, train_params, loader_params, model, start_optimizer, final_optimizer):
            print( "Epoch train stats: {}".format(epoch_stats) )


    @staticmethod
    def test_model(dataset: 'MNIST', loader_params: 'DataLoaderParams', model: Union[str, 'Path', 'Module']) -> None:
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
    train_dataset = Processing.get_mnist_dataset('./datasets/FashionMNIST_TRAIN', is_train=True)
    test_dataset = Processing.get_mnist_dataset('./datasets/FashionMNIST_TEST', is_train=False)

    train_params = TrainParams(epochs=10, lr=0.005, accuracy_threshold=0.9)
    train_loader_params = DataLoaderParams(batch_size=200, batch_shuffle=True)
    test_loader_params = DataLoaderParams(batch_size=len(test_dataset), batch_shuffle=False)

    model = Network()
    start_optimizer = Adam(model.parameters(), lr=train_params.lr, amsgrad=True)
    final_optimizer = SGD(model.parameters(), lr=train_params.lr, momentum=0.9, nesterov=True)

    Processing.train_model(train_dataset, train_params, train_loader_params, model, start_optimizer, final_optimizer)
    Processing.test_model(test_dataset, test_loader_params, model)




if __name__ == '__main__':
    main()
