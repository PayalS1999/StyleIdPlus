import torch
import torch.nn as nn


class MultiScaleDecoder(nn.Module):
    def __init__(self, base_decoder, low_channels, mid_channels, high_channels):
        """
        Args:
            base_decoder (nn.Module): The original decoder that can expose multi-scale features.
            low_channels (int): Number of channels in low-level features.
            mid_channels (int): Number of channels in mid-level features.
            high_channels (int): Number of channels in high-level features.
        """
        super(MultiScaleDecoder, self).__init__()
        self.base_decoder = base_decoder

        #low_channels = 320  # typically from the first block
        #mid_channels = 640  # from an intermediate block
        #high_channels = 1280  # from a deeper block

        # Injection modules: Convolution layers to fuse style features into decoder features.
        self.low_injection = nn.Conv2d(in_channels=low_channels, out_channels=low_channels, kernel_size=3, padding=1)
        self.mid_injection = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1)
        self.high_injection = nn.Conv2d(in_channels=high_channels, out_channels=high_channels, kernel_size=3, padding=1)

    def forward(self, latent, style_features):
        """
        Args:
            latent (Tensor): Latent representation obtained from the encoder.
            style_features (dict): Dictionary with keys 'low', 'mid', 'high' containing style feature maps.
        Returns:
            Tensor: The decoded (stylized) image.
        """
        # --- Low-Level Injection ---
        # Extract low-level features from the latent using the base decoder.
        low_features = self.base_decoder.get_low_features(latent)
        # Fuse low-level style features using the injection layer.
        low_features_injected = self.low_injection(low_features + style_features['low'])

        # --- Mid-Level Injection ---
        # Use the injected low-level features to compute mid-level features.
        mid_features = self.base_decoder.get_mid_features(low_features_injected)
        # Fuse the mid-level style features.
        mid_features_injected = self.mid_injection(mid_features + style_features['mid'])

        # --- High-Level Injection ---
        # Compute high-level features from the mid-level injected features.
        high_features = self.base_decoder.get_high_features(mid_features_injected)
        # Inject high-level style information.
        high_features_injected = self.high_injection(high_features + style_features['high'])

        # --- Final Decoding ---
        # Decode the high-level injected features to produce the final stylized image.
        output = self.base_decoder.decode(high_features_injected)
        return output