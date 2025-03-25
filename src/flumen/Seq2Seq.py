import torch
from torch import nn


class FFNet(nn.Module):
    """
    A fully connected neural network (feedforward network) used for learning mappings between input and output.
    This network consists of an input layer, several hidden layers, and an output layer. 
    It can be customized with different numbers of hidden layers, activation functions, and optional batch normalization.

    Args:
        in_size (int): The number of features in the input.
        out_size (int): The number of features in the output.
        hidden_size (list of int): A list of integers defining the number of neurons in each hidden layer.
        activation (nn.Module): The activation function to use in the hidden layers (default: ReLU).
        use_batch_norm (bool): Whether to use batch normalization after each hidden layer (default: False).
    """

    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size,
                 activation=nn.Tanh,            # try | nn.ReLU --- not good!
                 use_batch_norm=False):
        super(FFNet, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_size, hidden_size[0]))

        if use_batch_norm:
            self.layers.append(nn.BatchNorm1d(hidden_size[0]))

        self.layers.append(activation())

        for isz, osz in zip(hidden_size[:-1], hidden_size[1:]):
            self.layers.append(nn.Linear(isz, osz))

            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(osz))

            self.layers.append(activation())

        self.layers.append(nn.Linear(hidden_size[-1], out_size))

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)

        return input



class ConvNet(nn.Module):
    """
    A simple 1D Convolutional Neural Network (CNN) for sequence or time-series data. 
    This model consists of two convolutional layers followed by a fully connected output layer. 
    The network can be customized with different numbers of filters, kernel sizes, and optional batch normalization.

    Args:
        in_size (int): The number of input features (channels).
        out_size (int): The number of output features.
        kernel_size (int): The size of the convolution kernel (default: 3).
        num_filters (int): The number of filters (channels) in the convolutional layers (default: 64).
        activation (nn.Module): The activation function to use in the layers (default: ReLU).
        use_batch_norm (bool): Whether to apply batch normalization after convolutional layers (default: False).
    """

    def __init__(self, in_size, out_size, kernel_size=3, num_filters=64, activation=nn.ReLU, use_batch_norm=False):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_size, num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding=1)
        self.fc1 = nn.Linear(num_filters, out_size)
        
        self.activation = activation
        self.use_batch_norm = use_batch_norm

        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(num_filters)
            self.bn2 = nn.BatchNorm1d(num_filters)

    def forward(self, x):
        if x.dim()==2: x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.activation(x)

        x = torch.max(x, dim=2)[0]  # Global max pooling
        x = self.fc1(x)
        return x


class SelfAttention(nn.Module):
    """
    A self-attention mechanism that allows the model to focus on different parts of the input sequence based on learned attention weights. 
    This helps the model capture relationships between different elements of the input sequence, which is important for tasks like 
    machine translation, text summarization, etc.

    Args:
        input_dim (int): The number of features in the input sequence.
        hidden_dim (int): The number of features in the hidden representation.
        output_dim (int, optional): The number of output features. Defaults to hidden_dim.
        activation (nn.Module): The activation function to use (default: ReLU).
        use_batch_norm (bool): Whether to apply batch normalization (default: False).
    """

    def __init__(self, input_dim, output_dim, hidden_dim, activation=nn.ReLU, use_batch_norm=False):
        super(SelfAttention, self).__init__()
        self.attn_weights = nn.Parameter(torch.rand(input_dim, hidden_dim))
        self.attn_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.output_dim = output_dim if output_dim==0 else hidden_dim  # Use hidden_dim if output_dim is None

        self.activation = activation
        self.use_batch_norm = use_batch_norm

        if use_batch_norm:
            self.bn = nn.BatchNorm1d(self.output_dim)

    def forward(self, x):
        attn_scores = torch.matmul(x, self.attn_weights) + self.attn_bias
        attn_weights = torch.softmax(attn_scores, dim=-1)
        weighted_sum = torch.matmul(attn_weights, x)

        if self.use_batch_norm:
            weighted_sum = self.bn(weighted_sum)
        
        return self.activation(weighted_sum)


class ResidualBlock(nn.Module):
    """
    A residual block consisting of two fully connected layers with a skip connection (residual connection) 
    that allows gradients to flow more easily through the network. This helps mitigate the vanishing gradient problem 
    and allows for better learning in deeper networks.

    Args:
        input_dim (int): The number of input features.
        hidden_dim (int): The number of neurons in the hidden layer.
        output_dim (int): The number of output features.
        activation (nn.Module): The activation function to use (default: ReLU).
        use_batch_norm (bool): Whether to apply batch normalization (default: False).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, activation=nn.ReLU, use_batch_norm=False):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.activation = activation
        self.use_batch_norm = use_batch_norm

        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.activation(x)

        x = self.fc2(x)
        if self.use_batch_norm:
            x = self.bn2(x)

        return self.activation(x + residual)


class GRUEncoderDecoder(nn.Module):
    """
    A GRU-based encoder-decoder model where the encoder processes an input sequence, 
    and the decoder generates an output sequence. The output is passed through a fully connected layer 
    to match the desired output size. This model can be used for sequence-to-sequence tasks such as machine translation, 
    time-series forecasting, etc.

    Args:
        input_size (int): The number of input features for the encoder.
        hidden_size (int): The number of features in the hidden state of the GRU.
        output_size (int): The number of output features (the final output size of the decoder).
        activation (nn.Module): The activation function to use (default: ReLU).
        use_batch_norm (bool): Whether to apply batch normalization to the output (default: False).
    """

    def __init__(self, input_size, hidden_size, output_size, activation=nn.ReLU, use_batch_norm=False):
        super(GRUEncoderDecoder, self).__init__()
        self.encoder_gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.decoder_gru = nn.GRU(output_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.activation = activation
        self.use_batch_norm = use_batch_norm

        if use_batch_norm:
            self.bn = nn.BatchNorm1d(output_size)

    def forward(self, x):
        # Encoder
        _, h_n = self.encoder_gru(x)
        
        # Decoder
        output, _ = self.decoder_gru(h_n, h_n)
        
        output = self.fc(output)
        if self.use_batch_norm:
            output = self.bn(output)
        
        return self.activation(output)
