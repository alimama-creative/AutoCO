import numpy as np
import time

def train(train_queue, model, optimizer, arch_optimizer, logging):
	rewards_all = []
	losses_all = []
	for step, features in enumerate(train_queue):
		rewards = model.recommend(features)
		rewards_all.append(rewards)
		losses = model.step(optimizer, arch_optimizer, step)
		losses_all.append(losses)
		print("losses: ", losses, "rewards: ", rewards)
		print("model's log_alpha", model.log_alpha)
		logging.info("step: %s, rewards: %s"%(step, rewards))
		logging.info("step: %s, total_reward: %s" % (step, sum(rewards_all)))
	g, gp = model.genotype()
	return g, gp, np.mean(losses_all), sum(rewards_all)
