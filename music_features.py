import numpy as np
import scipy as sc
import madmom
import librosa



class Audio_features:
    def __init__(self, signal, fps):
        self.signal = signal
        self.fps = fps
        #self.num_frames = np.round(len(self.signal) * self.fps / self.signal.sample_rate) + 1
        fs = madmom.audio.signal.FramedSignal(self.signal, frame_size=2048, fps=self.fps)
        self.stft = madmom.audio.stft.STFT(fs)

    def log_filt_spectrogram(self):
        spec = madmom.audio.spectrogram.Spectrogram(self.stft)
        log_filt_spec = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(spec)
        return log_filt_spec

    def onset(self, log_filt_spec):
        superflux = madmom.features.onsets.superflux(log_filt_spec)
        onset = superflux / superflux.max()
        return onset

    def beats(self):
        proc = madmom.features.downbeats.RNNDownBeatProcessor()
        res = proc(self.signal, fps=self.fps)
        prob_beats = res[:, 0]
        prob_downbeats = res[:, 1]

        proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=self.fps)
        times = proc(res)
        beat_times = times[:, 0]
        downbeat_times = times[:, 1]

        return prob_beats, prob_downbeats, beat_times, downbeat_times

    def chroma(self):
        # dcp = madmom.audio.chroma.DeepChromaProcessor()
        # chroma = dcp(self.signal, fps = self.fps)
        #clp = madmom.audio.chroma.CLPChromaProcessor(fps = self.fps)
        #chroma = clp(self.signal)

        chroma = librosa.feature.chroma_stft(self.signal, sr = self.signal.sample_rate, S = abs(self.stft).T )

        return chroma.transpose(1,0)


if __name__ == "__main__":
    x = madmom.audio.signal.Signal("music_wav\mamacita.wav", sample_rate=None,
                                   num_channels=1, channel=None, dtype="float32")

    a = Audio_features(x, 25)

    c = a.chroma()
    print(c.shape)
    print(a.log_filt_spectrogram().shape)