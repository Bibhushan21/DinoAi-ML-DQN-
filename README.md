# Dino AI with DQN 🦖

This project implements an AI agent using **Deep Q-Networks (DQN)** to play the **Google Chrome Dino game**. The AI learns through reinforcement learning to optimize its performance and achieve high scores by dodging obstacles in real time.

## 🧠 Project Overview

The AI agent is trained using **DQN** to interact with the game environment, learning from its actions and improving over time. The AI observes the game screen, processes the visual input, and makes decisions (jump or duck) to avoid obstacles such as cacti and flying pterodactyls.

## 🔧 Tech Stack

- **Language**: Python
- **Libraries**:
  - TensorFlow / PyTorch
  - OpenCV (for image processing)
  - Gym (custom environment for the Dino game)
  - Numpy
  - Matplotlib (for visualizing training progress)

## 🚀 Getting Started
### best_model_125000.zip is the best trained model you can use while using this code 

### Prerequisites

Ensure you have Python installed. You'll also need to install the necessary libraries:

1. **Necessary libraries**:
   ```bash
   pip install tensorflow opencv-python gym numpy matplotlib


## Installation

1. **Clone the repo**:
   ```bash
   git clone https://github.com/bibhushan-subedi/dino-ai-dqn.git


2. **Run the training script**:
   ```bash
   python train_dino_ai.py

3. **Play the game using the trained AI**:
   ```bash
   python play_dino_ai.py

🛠️ Features
- **Deep Q-Network (DQN)**: The AI learns the optimal strategy by maximizing cumulative rewards through Q-learning.
- **Real-Time Gameplay**: The AI plays the Dino game in real time, making decisions based on the visual input from the game.
- **Custom Gym Environment**: A custom Gym environment replicates the Dino game for the AI to train and play.
- **Performance Optimization**: Uses experience replay and target networks for better training stability and performance.

🎨 AI Learning Process
- The AI observes the game state through the screen capture.
- It chooses actions (jump, duck, or do nothing) based on its current policy.
- Rewards are given based on its performance (positive for surviving, negative for hitting obstacles).
- The Q-network updates its policy by minimizing the loss between predicted Q-values and actual rewards.

📊 Results
- **Training Performance**: The model trains over several thousand episodes, progressively improving its ability to dodge obstacles.
- **High Scores**: After training, the AI can achieve high scores by accurately jumping or ducking under obstacles at the right time.

🧑‍💻 Authors
- Bibhushan Subedi – @bibhushan-subedi

🤝 Acknowledgements
- OpenAI Gym
- Google Chrome Dino Game
- [DQN Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

