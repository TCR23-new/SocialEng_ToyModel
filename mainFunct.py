#
import torch
import pickle
import os
from DataGenerator import DataExtraction,TrainModel,Employee,DatasetBuilder
from TrainSocEngAgent import CompanyEnv,EpisodeQueue,Agent,GatherData,train_Agent,GAMMA,STORAGE_SIZE,EPI_Q,policy_Q


if __name__ == '__main__':
    # Define variables
    NumIters = 600 #800
    NumEpisodes = 50
    NumEmployees = 3 #6 # Note: To get convergence, you will need to increase NumIters whenever you increase the number of employees
    NumInstances = 1000
    latent_dim = 100
    num_actions = 3
    learning_rate = 1e-3
    batch_size =100
    switch_flag = False
    current_files = os.listdir()
    data_filename = 'SocialEngText.txt'
    if data_filename not in current_files:
        print('Generating the text data')
        DatasetBuilder(10000,data_filename)
    # Load the data
    phrases,scores = DataExtraction(data_filename)
    print('Extracted the data')
    # train the employee's sentiment classifier
    model,vctzr = TrainModel(phrases,scores)
    print('Trained the sentiment classifier')
    # Initialize the agent
    agent = Agent(NumEmployees,latent_dim,num_actions)
    opt = torch.optim.Adam(agent.parameters(),lr=learning_rate)
    # Initialize the environment
    env = CompanyEnv(NumEmployees,model,vctzr,phrases,scores)
    #
    # Start the training process
    trn_loss,trn_perfs,tst_perfs = [],[],[]
    all_policy_data = []
    all_episode_data = []
    print('Starting the agent training process')
    for iter in range(NumIters):
        if iter%1 == 0 and iter !=0:#200
            print('Iteration {}'.format(iter))
        if iter !=0 and iter%400 == 0 and switch_flag == True:
            print('switching the resp probs')
            env.reset_resp_probs()
        # gather data
        avg_trn_rew = GatherData(env,agent,NumEpisodes,printFlag=True)
        trn_perfs.append(avg_trn_rew)
        # extract episodes for use in the agent's training process
        episodes = []
        policy_data1 = []
        for i in range(min(NumInstances,EPI_Q.queue_size())):
            episodes.append(EPI_Q.dequeue())
        for i in range(policy_Q.queue_size()):
            policy_data1.append(policy_Q.dequeue())
        if iter%100 == 0 and iter != 0: # 100
            all_policy_data.append(policy_data1)
            all_episode_data.append(episodes)
        else:
            policy_data1 = []
        # train the agent
        tr_loss = train_Agent(agent,episodes,opt,batch_size)
        trn_loss.append(tr_loss)
        # print('------Test Case-------')
        avg_tst_rew = GatherData(env,agent,NumEpisodes,trainFlag = False)
        tst_perfs.append(avg_tst_rew)
        # print('----------------------')
    pickle.dump([trn_perfs,tst_perfs],open('PerfData_TestAgent.pickle','wb'))
    pickle.dump([all_policy_data,all_episode_data],open('PolicyData_EpisodeData_overTraining.pickle','wb'))
