# 1. IMPORTING and INSTALLING DEPENDENCIES


#!pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

#!pip install stable-baselines3[extra] protobuf==3.20.*

# MSS is used to capture the screen
from mss import mss 
#pydirectinput is used to control the keyboard
import pydirectinput
#opencv is used to process the images
import cv2
#numpy is used to manipulate the images
import numpy as np
#pytesseract(OCR) is used for game over extraction
import pytesseract

# import tesseract
# visualize the captured screen
from matplotlib import pyplot as plt
# time is used to time the game and time for pause

import time
#gym is used to create the environment
# ENVIROMENT COMPONENTS 

from gym import Env
from gym.spaces import Discrete, Box

from graphviz import Digraph




class dinoGame(Env):  
    
    
    # Set up the environment action and observation space
    def __init__(self):
        #Sub class model 
        super().__init__()
        #setup Space
        self.observation_space = Box(low=0, high=500, shape=(1,83, 100), dtype=np.uint8)
        self.action_space = Discrete(3)
        
        #Define extraction parameter for the game (Grab Screenshot )
        self.cap = mss()
        self.game_location = {'top': 450, 'left': 0, 'width': 800, 'height': 800}
        self.done_location = {'top': 600, 'left': 850, 'width': 850, 'height': 300}
        
        
         
         
    # Called to do somethinghs in the game 
    def step(self, action):
        # 0 = jump, 1 = duck(down), 2 = do nothing
        action_map = {
            0: 'up',
            1: 'down',
            2: 'nothing'
        }
        if action != 2:
            pydirectinput.press(action_map[action])
            
            #check if game over
        over, gameover_capture = self.game_over()
            #return the new observation
        new_observation = self.get_observation()
            #return the reward-we get point for each frame  
        reward = 1
        
        # Info
        info = {}
        
        return new_observation, reward, over, info
    
    
    
    
    #Visualize the game
    def render(self):
        # cv2.imshow('Game', self.get_observation()[0])
        cv2.imshow('Game',np.array(self.cap.grab(self.game_location))[:,:,:3])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()
            
            
    #Close the observation
    def close(self):
        cv2.destroyAllWindows()
        pass
    
    
    
    
    #Reset the game
    def reset(self):
        time.sleep(1)
        pydirectinput.click(x=400, y=400)
        pydirectinput.press('space')
        
        return self.get_observation()
    
    
    
    
    
    
    #get the part of the observation of the game that we want 
    def  get_observation(self):
        #get the screenshot of the game
        raw_img = np.array(self.cap.grab(self.game_location))[:,:,:3]
        
          #convert the image to grayscale
        grey_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        
        #resize the image
        resized_img = cv2.resize(grey_img, (100,83))
        
        #Add chaNNel
        channel_img = np.reshape(resized_img, (1, 83, 100))
        
        #return the image
        
        return channel_img
    
    
    ##2. BUILDING THE MODEL
    ##2.1 Creating the Environment
    
    
    #Game over text USING OCR (pytesseract)
    def game_over(self):
        gameover_capture = np.array(self.cap.grab(self.done_location))[:,:,:3]
  
         #Valid game over
        gameover_text = ['GAME', 'GAHM',]
        
         #Check if game over
        
        over = False
        res = pytesseract.image_to_string(gameover_capture)[:4]
        if res in gameover_text:
            over = True
        return over, gameover_capture
         
    
    


# env = dinoGame()
# score = 50


#env.reset()


## Testing the environment

env = dinoGame()
 
 #obs = env.get_observation()
 
 # #Testing the enviroment 
 
# for episode in range(2):
#     obs = env.reset()
#     over = False
#     score = 0
#     while not over:
        
#         obs, reward, over, info = env.step(env.action_space.sample())
#         score+=reward
#     print(f'Totalreward for episode {episode} is {score}')


## 3. TRAINING THE MODEL

# import os  for file path management
import os 
# import base Callback fro saving model 

from stable_baselines3.common.callbacks import BaseCallback
#Check enviroment 
from stable_baselines3.common.env_checker import check_env


from gym.wrappers.monitoring import video_recorder




env = dinoGame()
# check_env(env)


class TrainingCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, verbose=1):
        super(TrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls)) 
            self.model.save(model_path)

        return True
   

class CustomCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.rewards = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.timesteps = []
        self.episode_count = 0

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Log rewards, losses, and timesteps
        reward = self.locals.get('rewards', 0)
        self.rewards.append(reward)
        
        if 'episode_rewards' in self.locals:
            episode_reward = np.sum(self.locals['episode_rewards'])
            self.episode_rewards.append(episode_reward)
        
        if 'episode_lengths' in self.locals:
            episode_length = np.sum(self.locals['episode_lengths'])
            self.episode_lengths.append(episode_length)

    
        
        self.timesteps.append(self.num_timesteps)
        losses = self.locals.get('loss')
        self.losses.append(losses)

        if self.n_calls % self.check_freq == 0:
            # Save model checkpoint
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
            return True
        
        
        
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'       


 #import DQN algorithm	

from stable_baselines3 import DQN
from stable_baselines3 import PPO



# Create  the  DQN model


model = DQN('MlpPolicy', env, tensorboard_log= LOG_DIR , verbose=1 ,buffer_size= 100000, learning_starts= 100)

#PPO
#model = PPO('MlpPolicy', env, tensorboard_log=LOG_DIR, verbose=1)
 

## Train out model


#Seting best model
#model.load(r'C:\Users\subed\UE_AI_py\train\3actions_100000')
model=DQN.load(os.path.join(r'C:\Users\subed\UE_AI_py\MachineLearning_DinoAI\train\best_model_125000'))
#model_path = os.path.join('train', 'best_model_125000')
#model.load('train','best_model_125000')



for episode in range(10):
    obs = env.reset()
    over = False
    score = 50
    episode_actions = []
    while not over:
        action, _ = model.predict(obs)
        
        obs, reward, over, info = env.step(int(action))
        
        score+=reward
        episode_actions.append(action)

   
    print(f'Totalreward for episode {episode} is {score}')
   
   
   
   # Load the data from the provided files
episode_length = np.load(r'C:\Users\subed\UE_AI_py\MachineLearning_DinoAI\train\episode_length.npy')
rewards = np.load(r"C:\Users\subed\UE_AI_py\MachineLearning_DinoAI\train\reward.npy")
episode_reward = np.load(r"C:\Users\subed\UE_AI_py\MachineLearning_DinoAI\train\episode_reward.npy")
episode_reward_PPO = np.load(r"C:\Users\subed\UE_AI_py\MachineLearning_DinoAI\train\episode_reward_PPO.npy")
losses = np.load(r"C:\Users\subed\UE_AI_py\MachineLearning_DinoAI\train\loss.npy")



# Plot line graph for episode_reward over episodes
plt.figure(figsize=(15, 6))
plt.plot(episode_reward_PPO, color='red')
plt.title('Episode Reward Over Time with PPO')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.show()




# Calculate moving average for episode_reward
window_size = 10
moving_average = np.convolve(episode_reward_PPO, np.ones(window_size) / window_size, mode='valid')

# Plot moving average for episode_reward
plt.figure(figsize=(12, 6))
plt.plot(moving_average, color='red')
plt.title('Moving Average of Episode Reward with PPO')
plt.xlabel('Episode (window size 10)')
plt.ylabel('Total Reward (Moving Average)')
plt.grid(True)
plt.show()



# Plot line graph for episode_reward over episodes
plt.figure(figsize=(15, 6))
plt.plot(episode_reward, color='green')
plt.title('Episode Reward Over Time with DDQ')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.show()



# Calculate moving average for episode_reward
window_size = 10
moving_average = np.convolve(episode_reward, np.ones(window_size) / window_size, mode='valid')

# Plot moving average for episode_reward
plt.figure(figsize=(12, 6))
plt.plot(moving_average, color='green')
plt.title('Moving Average of Episode Reward with DQN')
plt.xlabel('Episode (window size 10)')
plt.ylabel('Total Reward (Moving Average)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Loss Over Time')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.grid(True)
plt.show()