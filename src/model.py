from .decoder.facedecoder import FaceDecoder
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(MyModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_data):
        # Pass the input data through the encoder
        encoded_data = self.encoder(input_data)

        # Pass the encoded data through the decoder
        decoded_data = self.decoder(encoded_data)

        return decoded_data



if __name__ == '__main__':
    # Create an instance of the encoder
    encoder = FaceEncoder()

    # Create an instance of the decoder
    decoder = FaceDecoder()

    # Create an instance of the model
    model = MyModel(encoder, decoder)

    # Create some dummy input data
    input_data = torch.randn(1, 3, 256, 256)

    # Pass the input data through the model
    output = model(input_data)

    # Print the shape of the output
    print(output.shape)