import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import pickle
import os
from DataGenerator import DataExtraction,TrainModel,Employee,DatasetBuilder #(num_examples)

# NOTE: This environment assumes that an employee remains compromised
def softmax_fnct(x):
    return np.exp(x)/np.sum(np.exp(x))

class CompanyEnv:
    def __init__(self,NumEmployees,model,vctzr,all_resp,resp_type,penalty=0.1):
        self.num_emp = NumEmployees
        self.all_resp = all_resp
        self.resp_type = resp_type
        self.status = np.zeros((self.num_emp,))
        self.resp_prob = []
        self.penalty = penalty
        self.tally = 0
        self.cases = []
        self.create_resp_probs2()
        self.model = model
        self.vctzr = vctzr
        self.emp = [Employee(self.model,self.vctzr,r) for r in self.resp_prob]
    def create_resp_probs(self):
        for i in range(self.num_emp):
            tmp_vect = []
            for sent_class in np.unique(self.resp_type):
                tmp_vect.append(softmax_fnct(np.random.rand(2,)))
            self.resp_prob.append(tmp_vect)
    def create_resp_probs2(self):
        self.cases = []
        for i in range(self.num_emp):
            tmp_vect = []
            specific_case = np.random.choice([0,1,2])
            self.cases.append(specific_case)
            # print('specific case {}'.format(specific_case))
            for sent_class in np.unique(self.resp_type):
                if specific_case == sent_class:
                    tmp_vect.append([0.1,0.9])
                else:
                    tmp_vect.append([0.9,0.1])
                # tmp_vect.append(softmax_fnct(np.random.rand(2,)))
            self.resp_prob.append(tmp_vect)
    def reset_resp_probs(self):
        self.resp_prob = []
        print('Previous Cases per employee: {}'.format(self.cases))
        self.create_resp_probs2()
        self.emp = [Employee(self.model,self.vctzr,r) for r in self.resp_prob]
        print('New Cases per employee: {}'.format(self.cases))
    def next_state(self,compromised_emps):
        for e in compromised_emps:
            self.status[e] = 1
        return self.status
    def env_reset(self):
        self.status = np.zeros((self.num_emp,))
        self.tally = 0
        return self.status
    def take_action(self,actions):
        # for each employee check there action
        compromised_emps = []
        uncompromised_emps = np.nonzero(self.status == 0)[0]
        for ix in uncompromised_emps:
            # get a message that is aligned with the sentiment
            chosen_type = actions[ix]
            idx = np.nonzero(self.resp_type == chosen_type)[0]
            chosen_phrase = np.random.choice(idx)
            emp_decision = self.emp[ix].makeDecision(self.all_resp[chosen_phrase])
            # emp_decision = self.emp.makeDecision(chosen_phrase,self.resp_prob[ix][chosen_type])
            if emp_decision == 1:
                compromised_emps.append(ix)
        # create the reward:
        avail_emps = self.num_emp - len(compromised_emps)
        if avail_emps != 0:
            reward = 0
        else:
            reward = self.num_emp - self.penalty*self.tally*self.num_emp
        # define the next state
        previous_state = self.status
        next_state = self.next_state(compromised_emps)
        timeout_criteria = (self.tally > self.num_emp*3)
        if (self.num_emp - np.sum(next_state)) == 0 or timeout_criteria == True:
            done = 1
        else:
            done = 0
            self.tally += 1
        return next_state,reward,previous_state,actions,done


class EpisodeQueue: # This will store the episodes completed by the agent
    def __init__(self,MAX_Q_SIZE):
        self.queue = []
        self.max_q_size = MAX_Q_SIZE
    def enqueue(self,x):
        if self.queue_size() < self.max_q_size:
            self.queue.append(x)
        else:
            print('The queue is full. You must remove elements from the queue first! \n')
    def dequeue(self):
        if self.isEmpty() != True:
            first_element = self.queue[0]
            self.queue.pop(0)
            return first_element
        else:
            print('There is nothing in the queue to take out! \n')
            return []
    def isEmpty(self):
        if len(self.queue) == 0:
            return True
        else:
            return False
    def queue_size(self):
        return len(self.queue)


class Agent(nn.Module):
    def __init__(self,num_emps,latent_dim,num_actions):
        super(Agent,self).__init__()
        self.num_emps = num_emps # PLACEHOLDER
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.main_comp = nn.Sequential(
        nn.Linear(num_emps,latent_dim),
        nn.ReLU(),
        nn.Linear(latent_dim,latent_dim)
        )
        self.value_comp = nn.Sequential(
        nn.Linear(latent_dim,latent_dim//2),
        nn.ReLU(),
        nn.Linear(latent_dim//2,1)
        )
        self.policy_comp = nn.Sequential(
        nn.Linear(latent_dim,num_emps),
        nn.ReLU(),
        nn.Linear(num_emps,self.num_actions*num_emps)
        )
        self.act = nn.Softmax(dim=2)
    def forward(self,x):
        rep_out = self.main_comp(x)
        val_out = self.value_comp(rep_out)
        policy_rep_out = self.policy_comp(rep_out)
        policy_rep_out = policy_rep_out.reshape(x.shape[0],self.num_emps,self.num_actions)
        policy_out = self.act(policy_rep_out)
        return val_out,policy_out

####### CREATE THE AGENT TRAINING RELATED METHODS
GAMMA = 0.99
STORAGE_SIZE = int(1e5)
EPI_Q = EpisodeQueue(STORAGE_SIZE)
policy_Q = EpisodeQueue(STORAGE_SIZE)
#
def GatherData(env,agent,NumEpisodes,trainFlag = True,printFlag=False):
    agent.eval()
    all_rewards = []
    for j in range(NumEpisodes):
        # define the epsiode storage
        tmp_episode = []
        tmp_policy_data = []
        # initialize the environment
        state1 = env.env_reset()
        done,total_reward,print_ctr = 0,0,0
        while done != 1:
            # get the output from the agent
            with torch.no_grad():
                val_out,pol_out = agent(torch.tensor(state1).float().reshape(1,-1))
            pol_out = pol_out[0,:,:].numpy()
            tmp_policy_data.append(pol_out)
            # generate the action for the agent
            tmp_actions = []
            for i in range(pol_out.shape[0]):
                # draw an action
                tmp_actions.append(np.random.choice([0,1,2],p=pol_out[i,:]))
                if trainFlag == False:
                    if j == 0 and print_ctr == 0:
                        # print(pol_out)
                        print_ctr = 1
            # get the outputs from the environment
            state1,reward,state0,actions,done = env.take_action(tmp_actions)
            total_reward += reward
            # store the state outputs
            tmp_episode.append([state1,reward,state0,actions,done])
        # store the episode data in the epsiode queue
        all_rewards.append(total_reward)
        if trainFlag == True:
            EPI_Q.enqueue(tmp_episode)
            policy_Q.enqueue(tmp_policy_data)
    if printFlag == True:
        print('Average reward: {}'.format(np.mean(all_rewards)))
    return np.mean(all_rewards)

def n_step_return(episodes):
    est_returns = []
    for e in episodes:
        tmp_v_targ = []
        for ix in range(len(e)):
            G = 0
            for i,j in enumerate(range(ix,len(e))):
                G += e[ix][1]*(GAMMA**i)
            tmp_v_targ.append(G)
        est_returns.append(tmp_v_targ)
    return est_returns


class TrainingData(Dataset):
    def __init__(self,episodes):
        self.episodes = episodes # PLACEHOLDER
        self.targ_v_per_episode = n_step_return(episodes)
        self.targ_v = []
        self.actions = []
        self.states = []
        for ix,e in enumerate(self.episodes):
            for iy,te in enumerate(e):
                # extract and organize the states
                self.states.append(te[2])
                # extract and organize the target values
                self.actions.append(te[3])
                # extract and organize the actions
                self.targ_v.append(self.targ_v_per_episode[ix][iy])
    def __len__(self):
        return len(self.states)
    def __getitem__(self,idx):
        out = {}
        out['state'] = self.states[idx]
        out['action'] = self.actions[idx]
        out['targ_v'] = self.targ_v[idx]
        return out
    def collate_fn(self,batch):
        data = list(batch)
        states = torch.stack([torch.tensor(x['state']) for x in data]).unsqueeze(0).float()
        actions = torch.stack([torch.tensor(x['action']) for x in data]).unsqueeze(0).int()
        targ_v = torch.tensor([torch.tensor(x['targ_v']) for x in data]).squeeze().float()
        return states,actions,targ_v

#
def train_Agent(agent,episodes,opt,batch_size):
    # loss = 0
    agent.train()
    data_buffer = TrainingData(episodes)
    loss_fn_value = nn.MSELoss()
    trn_loss,ctr = 0,0
    data_loader = torch.utils.data.DataLoader(data_buffer,batch_size=batch_size,collate_fn=data_buffer.collate_fn,shuffle=True,num_workers=0)
    for ix,batch1 in enumerate(list(data_loader)):
        ctr+=1
        states,actions,targ_v = batch1
        # get the output from the agent
        if len(states.shape) == 3:
            states = states.squeeze(0)
            actions = actions.squeeze(0)
        val_out,policy_out = agent(states)
        # compute the value loss
        val_loss_comp = loss_fn_value(val_out.reshape(-1,1),targ_v.reshape(-1,1))
        # compute the policy loss
        adv = targ_v.reshape(-1,1) - val_out.reshape(-1,1).detach()
        for q in range(policy_out.shape[0]):
            for j in range(policy_out.shape[1]):
                tmp_distr = torch.distributions.Categorical(logits = policy_out[q,j,:])
                log_prob = tmp_distr.log_prob(actions[q,j])
                if q == 0:
                    tmp_policy_loss = -(adv[q,0]*log_prob)
                else:
                    tmp_policy_loss += -(adv[q,0]*log_prob)
        policy_loss_comp = tmp_policy_loss/policy_out.shape[0] # For now we will simply normalize by the batch size
        loss = val_loss_comp + policy_loss_comp
        loss.backward()
        opt.step()
        opt.zero_grad()
        trn_loss += loss.item()
    trn_loss = trn_loss/ctr
    agent.eval()
    return trn_loss
