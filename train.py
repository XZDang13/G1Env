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

from tqdm import trange

from RLAlg.alg.ppo import PPO
from RLAlg.nn.steps import StochasticContinuousPolicyStep, ValueStep
from RLAlg.buffer.replay_buffer import ReplayBuffer, compute_gae

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
        cfg.scene.num_envs = 4096
        self.env = gymnasium.make("G1Walk-v0", cfg=cfg)

        self.env_nums, self.obs_dim = self.env.observation_space.shape

        self.action_dim = self.env.action_space.shape[1]
        self.device = self.env.unwrapped.device


        self.encoder = EncoderNet(self.obs_dim, [512, 512]).to(self.device)
        self.actor = ActorLearnNet(self.encoder.dim, self.action_dim, [512]).to(self.device)
        self.critic = ValueNet(self.encoder.dim, [512]).to(self.device)
        

        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters()), 
            lr=3e-4
        )

        self.steps = 25

        self.rollout_buffer = ReplayBuffer(self.env_nums, self.steps)

        self.rollout_buffer.create_storage_space("observations", (self.obs_dim,), torch.float32)
        self.rollout_buffer.create_storage_space("actions", (self.action_dim,), torch.float32)
        self.rollout_buffer.create_storage_space("log_probs", (), torch.float32)
        self.rollout_buffer.create_storage_space("rewards", (), torch.float32)
        self.rollout_buffer.create_storage_space("values", (), torch.float32)
        self.rollout_buffer.create_storage_space("dones", (), torch.float32)

        self.batch_keys = ["observations", "actions", "log_probs", "rewards", "values", "returns", "advantages"]
        
        self.obs = None

        self.epochs = 1000
        self.update_iteration = 5
        self.batch_size = self.env_nums * 10
        self.gamma = 0.99
        self.lambda_ = 0.95
        self.value_loss_weight = 0.5
        self.entropy_weight = 0.01
        self.max_grad_norm = 0.5
        self.clip_ratio = 0.2
        self.regularization_weight = 1e-4

        self.desired_kl = 0.01
        self.learning_rate = 1e-3

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
        self.encoder.eval()
        self.actor.eval()
        self.critic.eval()

        obs = self.obs
        for i in range(self.steps):
            obs = process_obs(obs)
            action, log_prob, value = self.get_action(obs)
            next_obs, reward, terminate, timeout, info = self.env.step(action)
            
            done = terminate | timeout

            record = {
                "observations": obs,
                "actions": action,
                "log_probs": log_prob,
                "rewards": reward,
                "values": value,
                "dones": done
            }

            self.rollout_buffer.add_records(record)

            obs = next_obs

        self.obs = obs
        obs = process_obs(obs)
        _, _, value = self.get_action(obs)
        returns, advantages = compute_gae(
            self.rollout_buffer.data["rewards"],
            self.rollout_buffer.data["values"],
            self.rollout_buffer.data["dones"],
            value,
            self.gamma,
            self.lambda_
            )
        
        self.rollout_buffer.add_storage("returns", returns)
        self.rollout_buffer.add_storage("advantages", advantages)

        self.encoder.train()
        self.actor.train()
        self.critic.train()

    def update(self, num_iteration:int, batch_size:int):
        for _ in range(num_iteration):
            for batch in self.rollout_buffer.sample_batchs(self.batch_keys, batch_size):
                obs_batch = batch["observations"].to(self.device)
                action_batch = batch["actions"].to(self.device)
                log_prob_batch = batch["log_probs"].to(self.device)
                value_batch = batch["values"].to(self.device)
                return_batch = batch["returns"].to(self.device)
                advantage_batch = batch["advantages"].to(self.device)

                feature_batch = self.encoder(obs_batch, False)
                policy_loss, entropy, kl_divergence = PPO.compute_policy_loss(self.actor, log_prob_batch, feature_batch, action_batch, advantage_batch, self.clip_ratio, self.regularization_weight)
 

                value_loss = PPO.compute_clipped_value_loss(self.critic, feature_batch, value_batch, return_batch, self.clip_ratio)
                
                loss = policy_loss + value_loss * self.value_loss_weight - entropy * self.entropy_weight

                '''
                if kl_divergence > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_divergence < self.desired_kl / 2.0 and kl_divergence > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
                '''

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def train(self):
        obs, info = self.env.reset()
        self.obs = obs
        for epoch in trange(self.epochs):
            self.rollout()
            self.update(self.update_iteration, self.batch_size)

        torch.save([self.encoder.state_dict(), self.actor.state_dict(), self.critic.state_dict()], "model.pth")
        

def main():
    trainer = Trainer()

    trainer.train()

    trainer.env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()