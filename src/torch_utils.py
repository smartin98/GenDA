import csv
import torch

class LossLoggerCallback:
    def __init__(self, filename):
        self.filename = filename
        self.losses = []
        self.recon_losses = []
        self.kld_losses = []

    def __call__(self, epoch, loss, recon_loss, kld_loss):
        self.losses.append(loss)
        self.recon_losses.append(recon_loss)
        self.kld_losses.append(kld_loss)

        # Save the losses to a CSV file
        with open(self.filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Loss', 'ReconLoss', 'KLDLoss'])
            for i in range(len(self.losses)):
                writer.writerow([i+1, self.losses[i], self.recon_losses[i], self.kld_losses[i]])
                
                
                
def spectral_loss(pred, truth):
    pred_real = torch.real(torch.fft.fft2(pred))
    pred_imag = torch.imag(torch.fft.fft2(pred))

    truth_real = torch.real(torch.fft.fft2(truth))
    truth_imag = torch.imag(torch.fft.fft2(truth))

    return 0.5*(torch.mean(torch.log10(1+(pred_real-truth_real)**2)) + torch.mean(torch.log10(1+(pred_imag-truth_imag)**2)))