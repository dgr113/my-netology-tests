# coding: utf-8

from itertools import starmap
from string import ascii_lowercase
from typing import Tuple, Dict, Generator
from dataclasses import dataclass, field, InitVar

from torch import Tensor, no_grad, device, cuda, max as torch_max  # type: ignore
from torch.nn import Module  # type: ignore
from torch.nn.functional import cross_entropy, softmax  # type: ignore
from torch.optim import Optimizer  # type: ignore
from torch.utils.data import DataLoader, Dataset  # type: ignore

from .helpers import TrainParams, DataLoaderParams, CustomTextDataset, TrainStats

TORCH_DEVICE = device( 'cuda' if cuda.is_available() else 'cpu' )  # USE CUDA GPU




@dataclass
class TrainContext:
    dataset: InitVar['Dataset']

    train_params: 'TrainParams'
    loader_params: 'DataLoaderParams'

    model: 'Module'
    optimizer: 'Optimizer'

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

        if self._current_epoch < self.train_params.epochs:
            self._epoch_start()

            train_loss = 0
            train_passed = 0

            for X_batch, y_batch in Processing.loader_to_device(self._data):
                X_predicted = self.model(X_batch, y_batch)

                loss = cross_entropy(X_predicted, y_batch.flatten(0))
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_passed += 1

            print( "Train loss: {:.3f}".format(train_loss / train_passed) )
            epoch_stats = self._get_epoch_stats()
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
    def _batch_to_device(X: 'Tensor', y: 'Tensor') -> Tuple['Tensor', 'Tensor']:
        return X.to(TORCH_DEVICE), y.to(TORCH_DEVICE)

    @staticmethod
    def loader_to_device(data: 'DataLoader') -> Generator[Tuple['Tensor', 'Tensor'], None, None]:
        yield from starmap(Processing._batch_to_device, data)

    @staticmethod
    def train_model(dataset: 'Dataset', train_params: 'TrainParams', loader_params: 'DataLoaderParams', model: 'Module', optimizer: 'Optimizer') -> None:
        for epoch_stats in TrainContext(dataset, train_params, loader_params, model, optimizer):
            # print( "Epoch train stats: {}".format(epoch_stats) )
            pass

    @staticmethod
    def test_model(dataset: 'Dataset', loader_params: 'DataLoaderParams', model: 'Module') -> float:
        model.train()
        total, correct = 0, 0
        with no_grad():
            for images, labels in Processing.loader_to_device(DataLoader(dataset, batch_size=loader_params.batch_size, shuffle=loader_params.batch_shuffle)):
                outputs = model(images)
                _, predicted = torch_max(outputs.data, 1)
                total += labels.size()[0]
                correct += int( outputs.argmax(dim=1).eq(labels).sum().item() )

        result_accuracy = 100 * correct / total
        print( "Accuracy of the network on test data is: {}%".format(result_accuracy) )
        return result_accuracy


    @staticmethod
    def _predict_step(int2char: Dict[int, str], outputs: 'Tensor') -> str:
        outputs = outputs[-1]  # Get last doc, because there's only one of them

        # print(outputs.shape)  # [1, 5, 28] - 1 документ, 5 символов, 28 вероятностей следующег осимвола для каждого класса
        prob = softmax(outputs, dim=0).data
        _, result = torch_max(prob, dim=0)

        predicted_ind = int(result.item())
        predicted_char = int2char.get(predicted_ind, ' ')
        return predicted_char


    @staticmethod
    def predict(model: 'Module', int2char: Dict[int, str], char2int: Dict[str, int], chars: str) -> None:
        model.eval()

        with no_grad():
            for i in range(30):
                chars = "".join(chars)
                X = CustomTextDataset.get_tensor_data(char2int, chars, as_unsqueeze=True)
                outputs, _ = model(X)
                predicted_char = Processing._predict_step(int2char, outputs)
                chars += predicted_char

            result = ''.join(chars)
            print('Predicted: ', result)


    @staticmethod
    def predict_caesar_test(model: 'Module', int2char: Dict[int, str], char2int: Dict[str, int], chars: str) -> None:
        chars_ = list(chars)

        model.eval()
        with no_grad():
            last_char = chars_.pop()
            last_char_ind = ascii_lowercase.index( last_char.lower() )

            X = CustomTextDataset.get_tensor_data(char2int, chars_, as_unsqueeze=True)

            print("BEFORE")
            outputs, _ = model(X, None)
            print("AFTER")

            predicted_char = Processing._predict_step(int2char, outputs)
            predicted_char_ind = ascii_lowercase.index( predicted_char.lower() )

            diff_ind = predicted_char_ind - last_char_ind

            #######
            results = []
            for ch in chars:
                try:
                    correct_ind = ascii_lowercase.index( ch.lower() ) + diff_ind
                    real_ind = correct_ind if correct_ind < len(ascii_lowercase) else 0
                except Exception:
                    real_ind = None

                real_ch = ascii_lowercase[real_ind] if real_ind is not None else ' '
                results.append(real_ch)

            print(results)
            #######
