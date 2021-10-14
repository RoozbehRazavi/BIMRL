import numpy
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load environment

env = utils.make_env(args.env, args.seed)
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir, device, args.argmax, use_rim=args.use_rim)
print("Agent loaded\n")

# Run the agent

if args.gif:
    from array2gif import write_gif

    frames = []

# Create a window to view the environment
env.render('human')

for episode in range(args.episodes):
    obs = env.reset()
    done2 = False
    while True:
        env.render('human')
        if args.gif:
            frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))

        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done)

        if done or env.window.closed:
            if episode == 4:
                done2 = True
            break
    if done2:
        env.close()
        break
    # if env.window.closed:
    #    break
print('doneeee')
if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif + ".gif")
    print("Done.")
