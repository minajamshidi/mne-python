import numpy as np
from scipy import signal


class NoiseGenerator:
    """A class for generating noise signals."""

    def __init__(self, color):
        if color not in ["pink", "white"]:
            raise ValueError(
                f"The color {color} for noise type is not implemented yet."
                f" Valid values for the argument color are 'pink' and 'white'."
            )
        self.color = color

    @staticmethod
    def simulate_pink_noise(N, gamma=1):
        """
        Create a pink noise (1/f) with N points.

        (c) function copied from mne_hfo.simulate.

        Parameters
        ----------
        N: int
            Number of samples to be returned

        Returns
        -------
        pink_noise_array: numpy array
            1D array of pink noise
        """
        # todo: add gamma slope
        M = N
        # ensure that the N is even
        if N % 2:
            N += 1

        x = np.random.randn(N)  # generate a white noise
        X = np.fft.fft(x)  # FFT

        # prepare a vector for 1/f multiplication
        nPts = int(N / 2 + 1)
        n = range(1, nPts + 1)
        n = np.sqrt(n)

        # multiplicate the left half of the spectrum
        X[range(nPts)] = X[range(nPts)] / n

        # prepare a right half of the spectrum - a copy of the left one
        X[range(nPts, N)] = np.real(X[range(int(N / 2 - 1), 0, -1)])
        X[range(nPts, N)] -= 1j * np.imag(X[range(int(N / 2 - 1), 0, -1)])

        y = np.fft.ifft(X)  # IFFT

        y = np.real(y)
        # normalising
        y -= np.mean(y)
        y /= np.sqrt(np.mean(y**2))
        # returning size of N
        if M % 2 == 1:
            y = y[:-1]
        return y

    def noise_signal(self, n_samples):
        """
        Generate a noise signal with the given color and number of samples.

        generating a noise signal.
        :param n_samples:
        :return:
        """
        # todo: add channels?
        if self.color == "pink":
            return self.simulate_pink_noise(n_samples)
        elif self.color == "white":
            return np.random.randn(n_samples)


class SignalGenerator:
    """A class for generating different types of signals."""

    @staticmethod
    def narrowband_oscillation(f1, f2, sfreq, n_samples, n_sig=1):
        """
        Generate a random narrow-band signal resembling a narrow-band oscillation.

        :param f1:
        :param f2:
        :param sfreq:
        :param n_samples:
        :param n_sig:
        :return:
        """
        x1 = np.random.randn(n_sig, n_samples)
        b1, a1 = signal.butter(N=2, Wn=np.array([f1, f2]) / sfreq * 2, btype="bandpass")
        return signal.filtfilt(b1, a1, x1, axis=1)
