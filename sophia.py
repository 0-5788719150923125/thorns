import os

import lightning.pytorch as pl
from pytorch_optimizer import SophiaH
from torch import Tensor, nn, optim, utils
from torch.optim import Optimizer
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# define any number of nn.Modules (or use your current ones)
size1 = 20000
size2 = 28 * 28
size3 = 256
encoder = nn.Sequential(nn.Linear(size2, size1), nn.ReLU(), nn.Linear(size1, size3))
decoder = nn.Sequential(nn.Linear(size3, size1), nn.ReLU(), nn.Linear(size1, size2))


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)

        loss = nn.functional.mse_loss(x_hat, x)

        # important
        self.manual_backward(loss, create_graph=True)
        opt.step()

        self.log("train_loss", loss)

    def configure_optimizers(self):
        return SophiaH(self.parameters())


dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset)

autoencoder = LitAutoEncoder(encoder, decoder)

trainer = pl.Trainer(limit_train_batches=1000000, max_epochs=1, devices=[0])
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
