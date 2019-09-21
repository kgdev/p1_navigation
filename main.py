from dqn import learn
from unityagents import UnityEnvironment

env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")

learn(env)

#  learn(
#      env,
#      learning_starts=100000,
#      total_timesteps=100000,
#      exploration_fraction=0,
#      exploration_final_eps=0)

