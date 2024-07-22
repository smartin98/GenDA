import torch
import torch.nn as nn
import math

class MultiHeadDecoder(nn.Module):
    def __init__(self, latent_dim, output_channels, num_heads, output_size, hidden_dim=256):
        super(MultiHeadDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.output_size = output_size
        self.hidden_dim = hidden_dim

        # Shared initial layers, directly outputting the right shape for heads
        self.shared_layers = nn.Sequential(
            nn.Linear(latent_dim, num_heads * 4 * 4 * 512),
            nn.ReLU()
        )

        start_size = 16  
        upscale_factor = int(math.log2(output_size / start_size))
        if 2**upscale_factor != output_size / start_size:
            raise ValueError(f"Output size {output_size} must be a power of 2 multiple of the minimum size {start_size}")

        # Define individual decoder heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
                nn.ReLU(),

                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
                nn.ReLU(),

                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
                nn.ReLU(),

                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
                nn.ReLU(),

                nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1, bias=False),
            ) for _ in range(num_heads)
        ])


    
    def forward(self, z):
        # Pass through shared layers 
        z = self.shared_layers(z)

        # Reshape for decoder heads 
        z = z.view(-1, self.num_heads, 512, 4, 4)

        # Decode with each head
        outputs = [head(z[:, i, :]) for i, head in enumerate(self.heads)]

        # Concatenate head outputs along the channel dimension
        return torch.cat(outputs, dim=1) 

class MultiHeadEncoder(nn.Module):
    def __init__(self, latent_dim, input_channels, num_heads, input_size, hidden_dim):
        super(MultiHeadEncoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.num_heads = num_heads
        self.input_size = input_size
        self.hidden_dim = hidden_dim

        # Individual encoder heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, 32, 4, stride=2, padding=1, bias=False),
                nn.ReLU(),

                nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=False),
                nn.ReLU(),

                nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
                nn.ReLU(),

                nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
                nn.ReLU(),

                nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
                nn.ReLU(),
            ) for _ in range(num_heads)
        ])

        # Shared final layers, outputting the latent representation
        self.shared_layers = nn.Sequential(
            nn.Linear(num_heads * 512 * 4 * 4, 2*latent_dim) 
        )

    def forward(self, x):
        head_inputs = x.chunk(self.num_heads, dim=1) 

        head_outputs = [head(x) for x, head in zip(head_inputs, self.heads)]  

        # Flatten and concatenate head outputs
        x = torch.cat([h.view(h.size(0), -1) for h in head_outputs], dim=1) 

        # Pass through shared layers for final latent representation
        x = self.shared_layers(x)
        return x


class VAE(nn.Module):
    def __init__(self, input_channels, output_channels, latent_dim, hidden_dim=256, input_size=64, output_size=64, num_heads=3):
        super(VAE, self).__init__()
        self.encoder = MultiHeadEncoder(latent_dim, input_channels, num_heads, input_size, hidden_dim)
        self.latent_dim = latent_dim
        self.output_size = output_size
        
        self.decoder = MultiHeadDecoder(latent_dim, output_channels, num_heads, output_size, hidden_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu_logvar = self.encoder(x).view(-1, 2, self.latent_dim)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar