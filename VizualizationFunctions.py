# This script will contain plotting functions
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import numpy as np
from sklearn.feature_selection import f_classif,mutual_info_classif
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

def Feature_Scores_Plot(X,y):
    mi_scores = mutual_info_classif(X,y)
    mx_nrm_scores = mi_scores/np.max(mi_scores)
    plt.figure()
    plt.plot(range(1,X.shape[1]+1),mx_nrm_scores)
    plt.xlabel('Feature ID')
    plt.ylabel('Max Normalized Mutual Information')
    plt.grid(axis='y')
    return mx_nrm_scores

def CreateConfusionMatrix(y,ypred,labels):
    conf_mat = confusion_matrix(y,ypred)
    plt.figure()
    disp=ConfusionMatrixDisplay(conf_mat,display_labels=labels)
    disp.plot()
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('True Sentiment')
    return 0

def AgentPredictionPerEmployee(policy,sent_names):
    fig,ax = plt.subplots(nrows=int(np.ceil(policy.shape[0]/3)),ncols=3,sharey=True)
    for i in range(policy.shape[0]):
        ax[i].bar(range(policy.shape[1]),policy[i,:])
        ax[i].set_xticks(range(policy.shape[1]),sent_names)
        ax[i].set_xlabel('Sentiment Options')
        if i == 0:
            ax[i].set_ylabel('Logits')
        ax[i].set_title("Employee {}".format(i+1))
        ax[i].set_ylim([0,1])
        ax[i].grid(axis='y')
    fig.suptitle("Agent's predicted distribution per Employee")
    return 0

def PerfOverTraining(rlist,maxPerf):
    numIters = len(rlist)
    plt.figure()
    plt.plot(range(1,numIters+1),rlist)
    plt.plot(range(1,numIters+1),maxPerf*np.ones(numIters,),'--',c='k')
    plt.ylabel('Reward')
    plt.xlabel('Number of Training Iterations')
    plt.title("Agent's Performance over Training")
    plt.grid(visible=True)
    return 0

# Functions to visualize the environment
class CompanyViz:
    def __init__(self,num_employees):
        self.num_employees = num_employees
        self.emp_x_loc = []
        self.emp_y_loc = []
        self.board_fig,self.board_ax = None,None
        self.upper_ylim = 0
        self.emp_status = ['grey' for _ in range(self.num_employees)]
        self.sentiment_color = ['black' for _ in range(self.num_employees)]
    def create_square(self,xy_loc,wh,color,ax):
        ax.add_patch(mpl.patches.Rectangle(xy_loc,wh,wh,fill=True,facecolor=color,edgecolor='black',linewidth=2))
        return 0
    def employee_square(self,x,y,ax,sq1_color,sq2_color):
        s1,s2 = 1,0.5
        size_diff = s1 -s2
        offset = (s1-s2)/2
        self.create_square((x,y),s1,sq1_color,ax)
        self.create_square((x+offset,y+offset),s2,sq2_color,ax)
        return 0
    def create_board(self):
        self.board_fig, self.board_ax = plt.subplots()
        self.board_fig.suptitle('Company Visualizer',fontsize='xx-large')
        self.board_ax.set_title('Compromised by: Negative (red), Neutral (blue), Positive (green) Messages')
        self.upper_ylim =1+3*math.ceil(self.num_employees/3)
        self.board_ax.set_ylim([1,self.upper_ylim])
        self.board_ax.set_xlim([0,9])
        self.board_ax.set_xticks([])
        self.board_ax.set_yticks([])
        # create the separators
        t1,ctr,separator_y_coord = True,1,[]
        while t1 == True:
            if ctr < self.upper_ylim:
                separator_y_coord.append(ctr + 3)
                ctr += 3
            else:
                t1 = False
        for sy in separator_y_coord:
            self.board_ax.plot([0,9],sy*np.ones(2,),c='black')
        # create a square per employee
        locs = [1,4,7]
        base1 = 2
        for i in range(1,self.num_employees+1):
            self.employee_square(locs[i%3],base1,self.board_ax,self.sentiment_color[i-1],self.emp_status[i-1])#'black' 'grey'
            if i%3 == 0:# and i != 0
                base1 += 3
    def reset_board(self):
        self.board_fig,self.board_ax = None,None
        self.upper_ylim = 0
        self.emp_status = ['grey' for _ in range(self.num_employees)]
        self.sentiment_color = ['black' for _ in range(self.num_employees)]
        self.create_board()
    def update_board(self,status,sent_vect):
        for ix,s in enumerate(status):
            if s == 1:
                self.emp_status[ix] = 'white'
                tmp_score = sent_vect[ix]
                if tmp_score == 0:
                    self.sentiment_color[ix] = 'red'
                elif tmp_score == 1:
                    self.sentiment_color[ix] = 'blue'
                else:
                    self.sentiment_color[ix] = 'green'
        self.create_board()
    def episode_sequence(self,seq):
        # Note: This method assumes that the input is a list of tuples.
        # The 2 elements in the tuple are lists with the status and sentiment...
        # ... for a particular employee
        # Show the default board setup
        self.reset_board()
        self.board_fig.suptitle('Company Visualizer (Initial Setup)')
        for ix,s in enumerate(seq):
            a,b = s
            self.update_board(a,b)
            self.board_fig.suptitle('Company Visualizer (Iteration {})'.format(ix+1))
    def EpiSeqPackaging(self,policy_data,episode_data,trnPoint,epiID):
        recombine_data = []
        tmp_epi_poli = policy_data[trnPoint][epiID]
        tmp_epi_ = episode_data[trnPoint][epiID]
        for q in range(len(tmp_epi_)):
            #[state1,reward,state0,actions,done] are the elements in each list of tmp_epi_
            recombine_data.append((tmp_epi_[q][0],np.argmax(tmp_epi_poli[q],axis=1)))
        return recombine_data
#
# Example function calls
# NumEmployees = 9
# comp_viz = CompanyViz(NumEmployees)
# comp_viz.create_board()
# comp_viz.update_board([0,1,1,0,0,1,1,0,0],[-1,2,1,-1,-1,0,0,-1,-1])
