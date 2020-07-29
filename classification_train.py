import os
import madmom
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from music_features import Audio_features

import multiprocessing.dummy as multiprocessing


class Data:
    def __init__(self, audio_paths, sec=30, fps=25, bool_signal=False):
        self.audio_paths = audio_paths
        self.sec = sec
        self.fps = fps
        self.classes = {'blues': 0, 'classical': 1, 'country': 2,
                        'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6,
                        'pop': 7, 'reggae': 8, 'rock': 9}
        self.bool_signal = bool_signal

    def get_features(self, audio):
        audio_features = Audio_features(audio, fps=self.fps)
        log_filt_spec = audio_features.log_filt_spectrogram()
        onset = audio_features.onset(log_filt_spec)
        prob_beats, prob_downbeats, beat_times, downbeat_times = audio_features.beats()
        chroma = audio_features.chroma()

        onset = torch.tensor(onset).unsqueeze(1).float()
        prob_beats = torch.tensor(prob_beats).unsqueeze(1).float()
        prob_downbeats = torch.tensor(prob_downbeats).unsqueeze(1).float()
        chroma = torch.tensor(chroma).float()

        features = torch.cat((onset, prob_beats, prob_downbeats, chroma), 1).transpose(1, 0)
        return features

    def __getitem__(self, index):
        elem = self.audio_paths[index]
        genre = elem.split('.')[0]
        if self.bool_signal:
            x = madmom.audio.signal.Signal("genres/genres/" + genre + "/" + self.audio_paths[index], sample_rate=None,
                                           num_channels=1, channel=None, dtype="float32")

            x = x[:x.sample_rate * self.sec]
            features_x = self.get_features(x)
        else:
            df = pd.read_csv("genres/genres/" + genre + "/" + self.audio_paths[index], sep=',', header=None)
            features_x = df.values.transpose(1, 0)[:, :self.fps * self.sec]
        y = self.classes[genre]
        return features_x, y

    def __len__(self):
        return len(self.audio_paths)


class MusicalClassifierNet(nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, hidden_channels3, out_channels, classes):
        """
        MusicalClassifier
        :param in_channels:
        :param out_channels:
        :param classes:  number of music genres
        """
        super(MusicalClassifierNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, hidden_channels1, kernel_size=2)
        self.norm1 = nn.BatchNorm1d(hidden_channels1)
        self.drop1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(hidden_channels1, hidden_channels2, kernel_size=2)
        self.norm2 = nn.BatchNorm1d(hidden_channels2)
        self.drop2 = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv1d(hidden_channels2, hidden_channels3, kernel_size=2, stride=2)
        self.norm3 = nn.BatchNorm1d(hidden_channels3)
        self.drop3 = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(input_size=hidden_channels3, hidden_size=out_channels // 2,
                            batch_first=True, bidirectional=True)
        self.linear = nn.Linear(out_channels, classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.norm1(self.conv1(x)))
        output = self.drop1(output)
        output = self.relu(self.norm2(self.conv2(output)))
        output = self.drop2(output)
        output = self.relu(self.norm3(self.conv3(output)))
        output = self.drop3(output)
        output = output.transpose(1, 2)
        output, _ = self.lstm(output)
        output = self.linear(output[:, -1])
        # output = torch.mean(output, dim = 1)

        return output


class MusicalClassifier:
    def __init__(self, epoch, batch_size, train, val,
                 net_args={"hidden_channels1": 64, "hidden_channels2": 128, "hidden_channels3": 64, "out_channels": 32},
                 p_name="", fps=25, dropp_lr_epoch=[None], device='cpu', bool_signal=False):
        """
        MusicalClassifier
        :param epoch: number of epoch
        :param train: train data
        :param train: validation data
        :param dropp_lr_epoch: numbers of epoch when lr is dropped by 10
        """
        self.data_train = Data(train, fps=fps, bool_signal=bool_signal)
        self.trainloader = DataLoader(self.data_train, batch_size=batch_size, shuffle=True)

        self.data_val = Data(val, fps=fps)
        self.valloader = DataLoader(self.data_val, batch_size=1, shuffle=True)
        self.device = device
        self.p_name = p_name
        self.epoch = epoch
        self.classifier = MusicalClassifierNet(15, **net_args, classes=10)
        self.classifier = self.classifier.to(self.device)
        self.optimizer = optim.Adam(self.classifier.parameters(), 1e-3)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
        self.criterion = nn.CrossEntropyLoss()
        self.dropp_lr_epoch = dropp_lr_epoch

    def train(self):
        list_loss_v = []
        list_acc_v = []

        list_loss_t = []
        list_acc_t = []
        for ep in range(self.epoch):
            self.classifier.train()

            if ep in self.dropp_lr_epoch:
                self.scheduler.step()

            total_loss = 0.0
            total_correct = 0.0
            step = 0
            for x, y in self.trainloader:
                step += 1
                x = x.to(self.device).float()
                y = y.to(self.device)

                model_prob_labels = self.classifier(x)
                _, predict_labels = torch.max(F.softmax(model_prob_labels, -1), 1)

                loss = self.criterion(model_prob_labels, y)

                total_loss += loss.detach().item()
                total_correct += torch.sum(predict_labels == y.data)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_acc = total_correct / len(self.data_train)
            avg_loss = total_loss / step
            list_loss_t.append(avg_loss)
            list_acc_t.append(avg_acc)
            print("Epoch: {}, epoch_loss: {:.5}, accuracy: {:.5}".format((ep + 1), avg_loss, avg_acc))
            self.save(f'{p}/{self.p_name}/{ep}_classifier.pkl')

            # validation

            loss_val, acc_val = self.test(self.data_val, self.valloader)
            if len(list_acc_v) > 0:
                if list_acc_v[-1] < acc_val:
                    self.save(f'{p}/{self.p_name}/classifier.pkl')
            list_loss_v.append(loss_val)
            list_acc_v.append(acc_val)
            print("VALIDATION! Epoch: {}, loss: {:.5}, accuracy: {:.5}".format((ep + 1), loss_val, acc_val))

        self.save(f'{p}/{self.p_name}/end_classifier.pkl')
        return list_loss_t, list_acc_t, list_loss_v, list_acc_v

    def save(self, path):
        torch.save(self.classifier.state_dict(), path)

    def test(self, data, loader):
        self.classifier.eval()
        acc = 0.0
        loss = 0.0
        for x, y in loader:
            x = x.to(self.device).float()
            y = y.to(self.device)

            model_prob_labels = self.classifier(x)
            l = self.criterion(model_prob_labels, y)
            loss += l.detach().item()

            _, predict_labels = torch.max(F.softmax(model_prob_labels, -1), 1)
            acc += torch.sum(predict_labels == y.data)

        return loss / len(data), acc / len(data)


def train_net(arg_list):
    data_train, data_val, device_name, p_name, net_args, dropp_lr_epoch, bool_signal = arg_list
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    mus_class = MusicalClassifier(200, 512, data_train, data_val, net_args, p_name, fps=25,
                                  dropp_lr_epoch=dropp_lr_epoch,
                                  device=device, bool_signal=bool_signal)
    print(mus_class.classifier)
    list_loss_t, list_acc_t, list_loss_v, list_acc_v = mus_class.train()
    val = pd.DataFrame({"loss": list_loss_v, "acc": list_acc_v})
    val.to_csv(f'{p}/validation_{p_name}.csv', sep=',', index=False)

    tr = pd.DataFrame({"loss": list_loss_t, "acc": list_acc_t})
    tr.to_csv(f'{p}/train_{p_name}.csv', sep=',', index=False)

    test_loss, test_acc = test_net(f"{p}/{p_name}/classifier.pkl", data_test, 25, device, net_args)
    print(f"{device_name}, classifier.pkl: ", test_loss, test_acc)
    test_loss, test_acc = test_net(f"{p}/{p_name}/end_classifier.pkl", data_test, 25, device, net_args)
    print(f"{device_name}, end_classifier.pkl: ", test_loss, test_acc)


def test_net(path, test_data, fps, device, net_args):
    data = Data(test_data, fps=fps)
    loader = DataLoader(data, batch_size=1, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    classifier = MusicalClassifierNet(15, **net_args, classes=10)
    classifier = classifier.to(device)
    classifier.load_state_dict(torch.load(path))
    classifier.eval()

    acc = 0.0
    loss = 0.0
    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device)

        model_prob_labels = classifier(x)
        l = criterion(model_prob_labels, y)
        loss += l.detach().item()

        _, predict_labels = torch.max(F.softmax(model_prob_labels, -1), 1)
        acc += torch.sum(predict_labels == y.data)

    return loss / len(data), acc / len(data)


if __name__ == "__main__":
    data = []
    p = "exp1"
    bool_signal = False
    classes = ('blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock')
    for c in classes:
        names = os.listdir("genres/genres/" + c)
        data.extend(names)

    if not bool_signal:
        data = [d for d in data if d.split('.')[-1] == 'csv']
    else:
        data = [d for d in data if d.split('.')[-1] != 'csv']

    data_train, data_test = train_test_split(data, test_size=0.1)
    data_val, data_test = train_test_split(data_test, test_size=0.5)

    arg_list = [(data_train, data_val, 'cuda:0', '0',
                 {"hidden_channels1": 32, "hidden_channels2": 64, "hidden_channels3": 32, "out_channels": 32},
                 (50, 100, 150), bool_signal),
                (data_train, data_val, 'cuda:1', '1',
                 {"hidden_channels1": 16, "hidden_channels2": 24, "hidden_channels3": 32, "out_channels": 32},
                 (50, 100, 150), bool_signal),
                (data_train, data_val, 'cuda:0', '2',
                 {"hidden_channels1": 20, "hidden_channels2": 30, "hidden_channels3": 40, "out_channels": 40},
                 (50, 100, 150), bool_signal),
                (data_train, data_val, 'cuda:1', '3',
                 {"hidden_channels1": 64, "hidden_channels2": 128, "hidden_channels3": 64, "out_channels": 32},
                 (50, 100, 150), bool_signal)]

    mp = multiprocessing.Pool(processes=22)
    mp.map(train_net, arg_list)

    mp.close()
    mp.join()

    #train_net(arg_list[0], bool_signal)
