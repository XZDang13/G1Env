import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium
import torch
import torch.optim as optim

from RLAlg.nn.steps import StochasticContinuousPolicyStep, ValueStep

from model.encoder import EncoderNet
from model.actor import ActorLearnNet
from model.critic import ValueNet

from env.walk_env_cfg import G1WalkEnvCfg

def process_obs(obs):
    features = obs["policy"]
    return features

class Trainer:
    def __init__(self):
        cfg = G1WalkEnvCfg()
        cfg.scene.num_envs = 12
        self.env = gymnasium.make("G1Walk-v0", cfg=cfg)

        self.env_nums, self.obs_dim = self.env.observation_space.shape

        self.action_dim = self.env.action_space.shape[1]
        self.device = self.env.unwrapped.device

        self.encoder = EncoderNet(self.obs_dim, [512, 512]).to(self.device)
        self.actor = ActorLearnNet(self.encoder.dim, self.action_dim, [512]).to(self.device)
        self.critic = ValueNet(self.encoder.dim, [512]).to(self.device)

        encoder_params, actor_params, _ = torch.load("model.pth")
        self.encoder.load_state_dict(encoder_params)
        self.actor.load_state_dict(actor_params)

        self.encoder.eval()
        self.actor.eval()

        self.steps = 1000

        self.obs = None

    @torch.no_grad()
    def get_action(self, obs_batch:list[list[float]], determine:bool=False):
        obs_batch = self.encoder(obs_batch)
        actor_step:StochasticContinuousPolicyStep = self.actor(obs_batch)
        action = actor_step.action
        log_prob = actor_step.log_prob
        if determine:
            action = actor_step.mean
        
        critic_step:ValueStep = self.critic(obs_batch)
        value = critic_step.value

        return action, log_prob, value
    
    def rollout(self):
        obs = self.obs
        for i in range(self.steps):
            obs = process_obs(obs)
            action, log_prob, value = self.get_action(obs, True)
            #print(action)
            #action = torch.zeros(12, 23, device=self.device)

            next_obs, reward, terminate, timeout, info = self.env.step(action)
            print(reward)
            done = terminate | timeout

            if done.any():
                print("-----------")
            obs = next_obs

        self.obs = obs
        

    def test(self):
        obs, info = self.env.reset()

        self.obs = obs
        for epoch in range(10):
            self.rollout()
        

def main():
    trainer = Trainer()
    print(trainer.actor.policy.transform)
    print(trainer.actor.policy.max_action)

    trainer.test()

    trainer.env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()