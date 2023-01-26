import openai

bot :...
env = gym.make('LunarLander-v2')

n_games = 3
n_moves = 10

for i_game in range(n_games):
    # reset environment to the initial state
    observation = env.reset()

    for i_move in range(n_moves):
        
        env.render()

        # set random action 
        action = env.action_space.sample()

        #executing the action
        observation, reward, done, info = env.step(action)
        print("observation: ", observation)
        print("rewards: ", reward)
        if done:
            print("Game finished")
            break

env.close()

bot : ...
# if there is an error this will print
try:
    # if there are no more rows it will exit the loop and close
    if row == 0:
            csvNew.close()
            sys.exit()
    # read the data and get zillow's estimated value
    data = csvOld.row_values(row)
    zest = getZestimate(data[2])
    # write the new data in the row
    csvNew.writerow([data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], zest, data[9], data[10], data[11]])
    # print out what it is doing
    print("Processing row " + str(row))
    # increment
    row -= 1
# if there is an error this will print
except Exception as e:
        print("Error on row " + str(row))
        print(e)