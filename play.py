from dqn.dqn import AtariDDQN
import os
import sys

f_no_ext = os.path.splitext(sys.argv[1])[0]
dqn = AtariDDQN.load('%s' % f_no_ext, only_model=True, env_name='Pong-v0')
i_episode = int(f_no_ext.split("_")[-1])

#input_t = threading.Thread(target=wait_input)
#input_t.start()

rewards = []
for i in range(100):
    done = False
    r_sum = 0
    while not done:
        #dqns[i_model].env.render()
        _, reward, _, done = dqn.learning_step()
        r_sum += reward
    rewards.append(r_sum)
    print("Episode {} reward: {}".format(i, r_sum))

print("Avg. reward: {}".format(sum(rewards)/len(rewards)))