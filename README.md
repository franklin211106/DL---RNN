# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset



## DESIGN STEPS
### STEP 1: 
First, gather historical stock price data, such as daily closing prices, from a reliable source, ensuring the data is clean and continuous.

### STEP 2: 
Next, normalize the collected data, for example using MinMax scaling, and create sequences where a fixed number of past days are used to predict the next day’s price.


### STEP 3: 
Convert the sequences into the proper format for the RNN, typically as tensors with shape (batch_size, sequence_length, features), and organize them into training and testing datasets.


### STEP 4: 
Design the RNN architecture by selecting the type of recurrent layers (RNN, LSTM, or GRU), the number of layers and hidden units, and connect them to a fully connected output layer that predicts the stock price.


### STEP 5: 
Train the network on the training dataset using a suitable loss function, such as mean squared error, and an optimizer like Adam, while monitoring the loss to ensure the model learns effectively without overfitting.

### STEP 6: 
Finally, test the model on the unseen test data, compare the predicted stock prices with the actual prices, optionally reverse the normalization to get real values, and visualize the results to assess accuracy.




## PROGRAM

### Name:FRANKLIN F

### Register Number:212224240041

```python
 Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out






# Train the Model

def train_model(model, train_loader, criterion, optimizer, epochs=20):
    train_losses = []
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()          # Clear previous gradients
            outputs = model(x_batch)       # Forward pass
            loss = criterion(outputs, y_batch)  # Compute loss
            loss.backward()                # Backpropagation
            optimizer.step()               # Update weights

            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')
    return train_losses # Return the training losses

```

### OUTPUT

## Training Loss Over Epochs Plot
<img width="1388" height="469" alt="image" src="https://github.com/user-attachments/assets/44f1690b-79bf-4415-9fc2-9cae068cbe63" />

<img width="1183" height="681" alt="image" src="https://github.com/user-attachments/assets/991daae0-c7e8-4a35-b30d-4e4bb3b11f66" />


## True Stock Price, Predicted Stock Price vs time

<img width="1129" height="717" alt="image" src="https://github.com/user-attachments/assets/d4abaaa1-5d14-443d-a60f-7a977ad0f54e" />


### Predictions
<img width="408" height="120" alt="image" src="https://github.com/user-attachments/assets/13fbc234-9208-4a2c-9143-eedb7f9c87d5" />


## RESULT
Thus, we developed a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.
