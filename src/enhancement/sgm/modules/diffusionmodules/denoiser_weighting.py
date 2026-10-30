import torch


class EpsWeighting:
    def __call__(self, sigma):
        return sigma**-2
