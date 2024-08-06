import torch
import torch.nn.functional as F
import torch.distributions as dist

class DropoutAugmentation(object):
    def __init__(self, intensity: float):
        """
        Initialize the DropoutAugmentation class with the dropout probability
        :param p: (float) the dropout probability
        """
        self.p = intensity

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout to the input image
        :param img: input image
        :return: image after applying dropout
        """
        # Apply dropout to the image
        new_img = F.dropout(img, p=self.p, training=True)
        return new_img


class GaussianBlur(object):
    def __init__(self, p: float):
        """
        Initialize the GaussianBlur class with standard deviation of the gaussian noise
        :param p: (float) the standard deviation of the gaussian noise
        """
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian blur to the input image
        :param img: input image
        :return: image after applying gaussian blur
        """
        # create new tensor with same shape as img and random noise with mean 0 and std = self.p
        # this one results in the error 'TypeError: random_() received an invalid combination of arguments - got (std=float, mean=int, )'
        # new_img = torch.tensor(img) + torch.tensor(img).new(img.shape).random_(mean=0, std=self.p)
        # this one works
        new_img = img.clone().detach().requires_grad_(True) + img.clone().detach().new(
            img.shape
        ).normal_(mean=0, std=self.p)
        # Clamp the image pixel values between 0 and 1
        return torch.clamp(new_img, min=0)


class UniformBlur(object):
    def __init__(self, p: float):
        """
        Initialize the class with a float value p, which denotes the strength of the noise that is added.
        :param p: the strength of the noise that will be added.
        """
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        This function applies a uniform noise on the input image
        :param img: input image
        :return: returns the blurred image
        """
        # Create a new tensor of the same shape and dtype as the input image, and fill it with random noise
        # with uniform distribution between -p/2 and p/2
        new_img = img.clone().detach().requires_grad_(True) + img.clone().detach().new(
            img.shape
        ).uniform_(-self.p / 2, self.p / 2)
        # Clamp the values of the pixels of the image to be between 0 and 1
        return torch.clamp(new_img, min=0)
    

class NegBinNoise(object):
    def __init__(self, intensity: float):
        """
        Initialize the class with a float value p, which denotes the strength of the noise that is added.
        :param intensity: the strength of the noise that will be added.
        """
        self.p = intensity

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        This function applies a negative binomial noise on the input image.

        For Negative Binomial:
        total_count (float or Tensor) non-negative number of negative Bernoulli trials to stop, although the distribution is still valid for real valued count
        probs (Tensor) Event probabilities of success in the half open interval [0, 1)
        logits (Tensor) Event log-odds for probabilities of success

        :param img: input image
        :return: returns the augmented image
        """
        # Set some value for the total count, will be normalized later
        total_count = 10

        # Set a random value for probs
        probs = torch.rand(img.size(), device=img.device)

        # Sample NegBin noise
        negbin_noise = dist.NegativeBinomial(total_count=total_count, probs=probs).sample().float()

        # Normalize to 10,000 counts and log1p transform the noise
        negbin_noise = negbin_noise / negbin_noise.sum(dim=1, keepdim=True) * 10000
        negbin_noise = torch.log1p(negbin_noise)

        # Scale the noise by the strength of the noise p and the standard deviation of the image
        negbin_noise = negbin_noise * self.p

        # Add the noise to the image
        new_img = img.clone().detach().requires_grad_(True) + negbin_noise

        # Clamp the values of the pixels of the image to be non-negative
        return torch.clamp(new_img, min=0)