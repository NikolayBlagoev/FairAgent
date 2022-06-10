from decimal import Decimal
import logging
import numpy as np
from random import uniform, randint
from typing import cast, List
from uri.uri import URI
from scipy.stats import chisquare
from sklearn.cluster import AgglomerativeClustering
from sklearn.pipeline import make_pipeline
from math import exp, tan, degrees,atan, log, sqrt, ceil, floor, pi
from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.issuevalue.Value import Value
from geniusweb.issuevalue.ValueSet import ValueSet
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.UtilitySpace import UtilitySpace
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory, ProfileInterface
)
from geniusweb.progress.ProgressRounds import ProgressRounds



class THEMIS_RUDE(DefaultParty):

    def __init__(self):
        super().__init__()
        self._best_bids_so_far = []
        self._prop_outcome = 2
        self._p = 0.7                                   # ODDS OF REPEATING A BID IN THE EXPLORATORY PHASE
        self._kindness=0
        self._prop_T4T = self._prop_outcome             #CAN Be DIFFERENT BUT I DON'T KNOW WHAT BEHAVIOUR IT WILL LEAD TO
        # BUT YOU CAN ALWAYS TRY TO SEE WHAT HAPPENS :)
        self._ensmall_kindness = 1
        self._k = 15                                    # WINDOW SIZE
        self._k_min = 10                                # WINDOW MIN_SIZE IF DYNAMIC WINDOWS ARE USED
        self._k_max = 20                                # WINDOW MAX_SIZE IF DYNAMIC WINDOWS ARE USED
        self._c1 = Decimal(0.7)
        self._c2 = Decimal(0.6)
        self._c3 = Decimal(0.4)
        self._update_in_fairness = True
        self._factor = 1
        self._opponent_profile: ProfileInterface = None
        self._final_strategy_start = 0.95
        self._reservation_utility = 0.5
        self._top_n_bids_for_window = 3
        self._profile_exchanged = False
        self._alpha = Decimal(0.5)
        self._kindness = 0
        self._kindness_threshold_to_concede = 0.35       # GIVEN IN ABSOLUTE TERMS SO NON-NEGATIVE
        self._x_train = []
        self._connection_matrix = []
        self._new_windows = []
        self._window_1 = []
        self._last_offered = None
        self._window_2 = []
        self._window_3 = []
        self._total_est: dict[str,dict[str, float]] = {}
        self._weight_est: dict[str,float] = {}
        self._init_flag = True
        self._history = []
        self._window_history = []
        self._window_progresses=[]
        self._utarget_history = []
        self._prefer_bits = []
        self._bids_set: set = {None}
        self._state = 0
        self._u_target_max=0.88
        self._utarget = self._u_target_max
        self._cmax = 0.06/self._prop_outcome            # MAXIMUM CONCESSION EACH ROUND
        self._flag_window_update = False
        self.getReporter().log(logging.INFO, "party is initialized")
        self._profile: ProfileInterface = None
        
        self._flag = True
        self._last_received_bid: Bid = None

    def notifyChange(self, info: Inform):
        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(info, Settings):
            self._settings: Settings = cast(Settings, info)
            self._me = self._settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self._progress: ProgressRounds = self._settings.getProgress()

            # the profile contains the preferences of the agent over the domain
            self._profile = ProfileConnectionFactory.create(
                info.getProfile().getURI(), self.getReporter()
            )
            
            #self._opponent_profile = ProfileConnectionFactory.create(
            #   URI("file:PUT_URI_HERE"), self.getReporter()
            #)
        # ActionDone informs you of an action (an offer or an accept) 
        # that is performed by one of the agents (including yourself).
        elif isinstance(info, ActionDone):
            action: Action = cast(ActionDone, info).getAction()
            actor = action.getActor()

            # ignore action if it is our action
            if actor != self._me:
                # if it is an offer, set the last received bid
                if isinstance(action, Offer):
                    self._last_received_bid = cast(Offer, action).getBid()
        # YourTurn notifies you that it is your turn to act
        elif isinstance(info, YourTurn):
            # execute a turn
            self._myTurn()

            # log that we advanced a turn
            self._progress = self._progress.advance()

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(info, Finished):
            # terminate the agent MUST BE CALLED
            self.terminate()
        else:
            self.getReporter().log(
                logging.WARNING, "Ignoring unknown info " + str(info)
            )

    # lets the geniusweb system know what settings this agent can handle
    # leave it as it is for this course
    def getCapabilities(self) -> Capabilities:
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )
    
    # terminates the agent and its connections
    # leave it as it is for this course
    def terminate(self):
        self.getReporter().log(logging.INFO, "party is terminating:")
        
        super().terminate()
        if self._profile is not None:
            self._profile.close()
            self._profile = None
            self._best_bids_so_far = []
            self._k = 15
            self._flag = True
            self._window_1 = []
            self._window_2 = []
            self._window_3 = []
            self._utarget = 0.94
            

    #######################################################################################
    ########## THE METHODS BELOW THIS COMMENT ARE OF MAIN INTEREST TO THE COURSE ##########
    #######################################################################################

    # give a description of your agent
    def getDescription(self) -> str:
        return "THEMIS agent by Nikolay Blagoev"
    
    def _set_u_target(self)->Decimal:
        copy_s = self._utarget
        profile = self._profile.getProfile()
        ur = []
        for bid in self._window_2:
            ur.append(profile.getUtility(bid))
        ur.sort()
        ur = ur[:4]
        # print(ur)
        median = np.median(ur)
        

        ur2 = []
        for bid in self._window_1:
            ur2.append(profile.getUtility(bid))
        ur2.sort()
        ur2 = ur2[:4]
        # print(ur2)
        median2 = np.median(ur2)
        # print(list(map(lambda b: profile.getUtility(b),self._best_bids_so_far)))
        median_change = median-median2

        # PROPORTIONAL CONCESSION
        if median_change>0:
            median_change = median_change/Decimal(self._prop_T4T)
        
        
        delta = Decimal.min(Decimal(self._cmax), median_change)
        print("DELTA %f", delta)
        if self._utarget>self._u_target_max*self._prop_outcome and self._state<3:
            delta =Decimal.max(delta, Decimal((Decimal(self._utarget)-Decimal(self._u_target_max*self._prop_outcome))/(3-self._state)))
        # self._utarget=Decimal.min(Decimal(0.9),Decimal.max(profile.getUtility(self._best_bids_so_far[0]),Decimal(self._utarget)-delta))
        copy_s=Decimal.min(Decimal(self._u_target_max),Decimal(self._utarget)-delta)
        # print("UTIL_TARGET %f"%self._utarget)
        
        return copy_s


    # execute a turn
    def _myTurn(self):
        if self._init_flag:
            domain = self._profile.getProfile().getDomain()
            for issue in domain.getIssues():
                self._weight_est[issue] = 0
            
            self._init_flag = False
        # check if the last received offer is good enough
        if self._last_received_bid is not None:
            profile = self._profile.getProfile()
            self._history.append(self._last_received_bid)
            self._value_estimate()
            self._best_bids_so_far.append(self._last_received_bid)
            self._best_bids_so_far.sort(key = lambda b : profile.getUtility(b), reverse=True)
            self._best_bids_so_far = self._best_bids_so_far[:10]
                
            self._window_3.append(self._last_received_bid)
            if len( self._window_3)== self._k:
                self._utarget_history.append(self._utarget)
                self._window_progresses.append(self._progress.get(0))
                self._window_history.append(self._window_3.copy())
                self._window_1 = self._window_2
                self._window_2 = self._window_3
                self._window_3 = []
                if len(self._window_1) == self._k:
                    # print("ARRAYS: %d %d"%(len(self._window_1), len(self._window_2)))
                    out = self._set_u_target()
                    self._flag_window_update = True
                    out2 = self._weight_estimator()
                    if out2 !=0 or self._opponent_profile is not None:
                        self._kindness_calculation()
                        self._state +=1
                    # print("two us %f %f"%(out, self._utarget))
                    
                    chance_p = uniform(0,1)<((abs(float(self._kindness))-self._kindness_threshold_to_concede)/(1-self._kindness_threshold_to_concede))
                    if self._kindness>self._kindness_threshold_to_concede and chance_p:
                        self._utarget-=Decimal(self._cmax/self._prop_outcome)
                        self._utarget=Decimal.max(Decimal(0.1),self._utarget)
                        print("IGNORING THEIR CONCESSION! I AM GOING DOWN!")
                    elif self._kindness<-self._kindness_threshold_to_concede and chance_p:
                        print("IGNORING THEIR CONCESSION! I AM GOING UP!")
                        self._utarget+=Decimal(self._cmax/self._prop_outcome)
                        self._utarget=Decimal.min(Decimal(self._u_target_max),self._utarget)
                    else:
                        if self._utarget<=out and out2!=0:
                            self._utarget=Decimal(self._utarget)-Decimal(out2)/(Decimal(1)*Decimal(len(self._profile.getProfile().getDomain().getIssues())))
                            print("They conceded but just not well enough for us")
                        else:
                            self._utarget=out
                    print("2: UTIL TARGET NEW %f"%self._utarget)
                    if self._state == 0:
                        self._state = 1
        if self._shouldaccept(self._last_received_bid):
            # if so, accept the offerr
            print("2: ACCEPTING %f"%self._op_profile(self._last_received_bid))
            action = Accept(self._me, self._last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            bid = self._findBid()
            self._last_offered=bid
            action = Offer(self._me, bid)

        # send the action
        self.getConnection().send(action)

    # method that checks if bid is acceptable given our current windows
    def _isGood(self, bid: Bid) -> bool:
        if bid is None:
            return False
        profile = self._profile.getProfile()

        return profile.getUtility(bid) > self._utarget and profile.getUtility(bid) < self._utarget+Decimal(0.3)

    def _eval_bid(self, bid: Bid, profile)->Decimal:
        us = profile.getUtility(bid)
        op = Decimal(self._op_profile(bid))
        return self._c1*profile.getUtility(bid)-self._c2*self._fairness_function(us,op,0)-self._c3*self._dynamic_fairness(us,op,0)
    
    def _shouldaccept(self, bid: Bid) -> bool:
        if bid is None:
            return False
        profile = self._profile.getProfile()
        if self._state<4:
            # Acceptance strategy for Exploratory and Setup T4T
            if self._c1>self._c2+self._c3:
                return self._isGood(bid)
           
            return False
        if self._progress.get(0) > self._final_strategy_start:
             # Acceptance strategy for Eager
            if self._c1>self._c2+self._c3:
                if len(self._best_bids_so_far)>0 and profile.getUtility(bid)>self._utarget:
                    if profile.getUtility(bid)>profile.getUtility(self._best_bids_so_far[0]):
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                if self._last_offered is None:
                    return False
                else:
                    return self._eval_bid(self._last_offered,profile)<self._eval_bid(bid,profile)
        # Acceptance for fairness optimizer:
        if(profile.getUtility(bid)<self._utarget):
            return False
        eval_bid = self._eval_bid(bid, profile)
        if(len(self._prefer_bits)==0):
            return False
        i = len(self._prefer_bits)-1
        count = 0
        # Accept if better than any of the next 3 bids we will propose:
        while i>=0 and count<3:
            if(eval_bid>=self._eval_bid(self._prefer_bits[i], profile)):
                return True
            i=i-1
            count+=1
        return False
    
    
        
    def _rand_Bid(self, last:Bid = None) -> Bid:
        domain = self._profile.getProfile().getDomain()
        all_bids = AllBidsList(domain)
        profile = self._profile.getProfile()
        while True:
            bid = all_bids.get(randint(0, all_bids.size() - 1))
            if self._isGood(bid):
                
                if bid in self._bids_set:
                    if uniform(0,1)<self._p:
                        if last is None:
                            break
                        elif (last is not None and profile.getUtility(last)>=profile.getUtility(bid)):
                            if uniform(0,1)<0.1:
                                break
                        else:
                            break
                else:
                    if last is None:
                        self._bids_set.add(bid)
                        break
                    elif (last is not None and profile.getUtility(last)>=profile.getUtility(bid)):
                        if uniform(0,1)<0.1:
                            self._bids_set.add(bid)
                            break
                    else:
                        self._bids_set.add(bid)
                        break
                    
        # print("BID %f "%self._profile.getProfile().getUtility(bid))
        return bid
    def _findBid(self) -> Bid:
        
        # compose a list of all possible bids
        profile = self._profile.getProfile()

        if self._progress.get(0) > self._final_strategy_start:
            print("PROGRESS")
            if self._c1>self._c2+self._c3:
                if len(self._best_bids_so_far) > 0: 
                    bid = self._best_bids_so_far[0]
                    self._best_bids_so_far = self._best_bids_so_far[1:]
                    if profile.getUtility(bid) > self._reservation_utility:
                        return bid 
                    
                return self._rand_Bid()
            else:
                if self._last_offered is None:
                    return self._rand_Bid()
                else:
                    return self._last_offered
        if self._state >= 1 and self._state<4:
            print("SECOND STRAT")
            # In this phase randomly give offers above u_min and below u_max which are hopefully lower than last
            self._prefer_bits = []
            return self._rand_Bid(self._last_offered)
        elif self._state == 0:
            if len(self._prefer_bits)==0:
                domain = self._profile.getProfile().getDomain()
                all_bids = AllBidsList(domain)
                n = all_bids.size()
                for i in range(n):
                    self._prefer_bits.append(all_bids.get(i))
                    self._prefer_bits.sort(key = lambda b : profile.getUtility(b), reverse=False)
                    if len(self._prefer_bits)>40:
                        self._prefer_bits = self._prefer_bits[1:]
                    # print( profile.getUtility(self._prefer_bits[0]))
            out = self._prefer_bits[len(self._prefer_bits)-1]
            if uniform(0,1)>self._p:
                self._prefer_bits.pop()
            return out
        else:
            print("Third strat")
            if self._flag_window_update:
                self._prefer_bits=[]
                self._flag_window_update = False
            if len(self._prefer_bits)==0:
                domain = self._profile.getProfile().getDomain()
                all_bids = AllBidsList(domain)
                n = all_bids.size()
                for i in range(n):
                    if not self._isGood(all_bids.get(i)):
                        continue
                    self._prefer_bits.append(all_bids.get(i))
                    self._prefer_bits.sort(key = lambda b : self._eval_bid(b,profile), reverse=False)
                    if len(self._prefer_bits)>self._k*4:
                        self._prefer_bits = self._prefer_bits[1:]
                    # print( profile.getUtility(self._prefer_bits[0]))
                # for b in self._prefer_bits:
                #     print("BID WITH ESTIMATED %f , %f" %(profile.getUtility(b),self._op_profile(b)))
            print(len(self._prefer_bits))
            out = self._prefer_bits.pop()
            print("2: SENDING %f , %f, Fairness: %f, dynamic: %f, eval: %f, target: %f, kindness: %f" %(profile.getUtility(out),self._op_profile(out),
            self._fairness_function(profile.getUtility(out),Decimal(self._op_profile(out)),0),
            self._dynamic_fairness(profile.getUtility(out),Decimal(self._op_profile(out)),0),
            self._eval_bid(out,profile),self._utarget, self._kindness))
            return out


    # FUNCTIONS RELATED TO THE FAIRNESS CALCULATIONS FOR AGENT THEMIS   
    def _kindness_calculation(self):
        new_kind = 0
        i = -1
        profile = self._profile.getProfile()
        prev_fairness = 0
        one_tick = self._window_progresses[1]-self._window_progresses[0]
        # print("%f IS ONE TICK %f %f %f"%(one_tick, self._window_progresses[0], self._window_progresses[1], self._window_progresses[2]))
        for window in self._window_history:
            i+=1
            if self._window_progresses[i]<0.4:
                continue
            acceptable_window_size = Decimal(1-exp((1-min(0.95,(self._window_progresses[i]-0.4)/(0.95-0.4)))**(2)*log(0.2)))
            kindness_window = []
            for bid in window:
                us_util = profile.getUtility(bid)
                opponent = self._op_profile(bid)
                fair_calc = 1-self._fairness_function(us_util,Decimal(opponent),Decimal(0))
                if (us_util<(self._utarget_history[i]-acceptable_window_size)):
                    
                    fair_calc=fair_calc*us_util/(self._utarget_history[i]-acceptable_window_size)
                kindness_window.append(fair_calc)
            kindness_window.sort(reverse=True)
            kindness_window=kindness_window[:3]
            median_fairness = np.median(kindness_window)
            
            delta = median_fairness-prev_fairness
            delta_ideal  = Decimal((1-prev_fairness)/Decimal(((0.95-self._window_progresses[i])/one_tick)))
            normalised_delta = delta-delta_ideal
            # print("%f is delta ideal but actual delta is %f for %f and prev %f curr_fairness %f and wind %f"
            # %(delta_ideal, delta, self._window_progresses[i], prev_fairness, median_fairness, acceptable_window_size))
            if normalised_delta > 0:
                normalised_delta = normalised_delta/(1-prev_fairness-delta_ideal)
            elif normalised_delta<0:
                normalised_delta = (normalised_delta)/(prev_fairness+delta_ideal)
            # print("NORM %f "%normalised_delta)
            if abs(normalised_delta)>0.05:
                new_kind = self._alpha*new_kind+(1-self._alpha)*normalised_delta
            # print("newkind %f "%new_kind)
                
            prev_fairness = median_fairness
        if self._opponent_profile is None:
            self._kindness = new_kind*Decimal(self._progress.get(0))
        else:
            self._kindness = new_kind
        # print("FINAL %f"%self._kindness)     


    
    def _dynamic_fairness(self, us: Decimal, opponent: Decimal, target: Decimal) -> Decimal:
        opponent = float(opponent)
        us = float(us)
        kindness_addition = float(self._kindness)
        if kindness_addition>0.2:
            # kindness=kindness-0.2
            # kindness=kindness/0.8
            # if kindness==1:
            #     return +pi/4
            kindness_addition = (kindness_addition-0.2)/0.8
            if kindness_addition == 1:
                kindness_addition = +pi/4
            else:
                kindness_addition= atan(1/(1-kindness_addition))-pi/4
        elif kindness_addition<-0.2:
            kindness_addition = (kindness_addition+0.2)/0.8
            kindness_addition=-atan(-1-kindness_addition)-pi/4
        else:
            kindness_addition=0
        prop = self._prop_outcome
        kindness_addition=kindness_addition/self._ensmall_kindness
        std = 0.1
        if opponent != 0:
            fraction = atan(us/opponent)
        else:
            fraction = 0
        
        out = degrees(abs(fraction-atan(prop)+kindness_addition))/90
        out=max(0,min(1,out))
        if (prop>=1 and fraction>atan(prop))or(prop<=1 and fraction<atan(prop)):
            
            # print("%f %f"%(e[0],e[1]))
            # less mean function:
            return Decimal(1-exp(-((prop)**self._factor)*(out/std)**2))
        else:
            # stricter function:
            return Decimal(1-exp(-(out/std)**2/((prop)**self._factor)))
    def _fairness_function(self, us: Decimal, opponent: Decimal, target: Decimal) -> Decimal:
        opponent = float(opponent)
        us = float(us)
        
        prop = self._prop_outcome
        factor = 1
        std = 0.1
        if opponent != 0:
            fraction = atan(us/opponent)
        else:
            fraction = 0
        if opponent == 0:
            out = degrees(abs(fraction-atan(prop)))/90
        else:
            out = degrees(abs(fraction-atan(prop)))/90
        if (prop>=1 and fraction>atan(prop))or(prop<=1 and fraction<atan(prop)):
            
            # print("%f %f"%(e[0],e[1]))
            # less mean function:
            return Decimal(1-exp(-((prop)**factor)*(out/std)**2))
        else:
            # stricter function:
            return Decimal(1-exp(-(out/std)**2/((prop)**factor)))
        
    # FUNCTIONS RELATED TO THE OPPONENT MODEL
    def _value_estimate(self):
        if self._opponent_profile is not None:
            
            return 0
        domain = self._profile.getProfile().getDomain()
        domain.getIssues()
        # total_est: dict[str,dict[Value, int]] = {}
        opponent_offers = self._history

        
        for issue in domain.getIssues():
            vl_est: dict[str, int] = {}
            maximal = -1
            n = domain.getIssuesValues().get(issue).size()
            for i in range(n):
                val = domain.getIssuesValues().get(issue).get(i).getValue()
                count = 1
                
                for bid in opponent_offers:
                    if(bid.getIssueValues().get(issue).getValue() == val):
                        count+=1
                if count>maximal:
                    maximal=count
                vl_est[val] = count
                
            for key in vl_est.keys():
                val = vl_est.get(key)
                val = val/maximal
                val = val**0.25
                vl_est[key] = val
            self._total_est[issue] = vl_est
        return

    def _weight_estimator(self, prog = None)->float:
        if self._opponent_profile is not None:
            wind1_op_val = []
            for b in self._window_1:
                wind1_op_val.append(self._opponent_profile.getProfile().getUtility(b))
            wind1_sum = np.median(wind1_op_val)
            wind2_op_val= []
            for b in self._window_2:
                wind2_op_val.append(self._opponent_profile.getProfile().getUtility(b))
            wind2_sum =  np.median(wind2_op_val)
            diff = Decimal.max(Decimal(0),wind1_sum-wind2_sum)
            print("DIFF %f"%diff)
            return diff
        unchanged = []
        # profile = self._profile.getProfile()
        progress = self._progress.get(0)
        if prog is not None:
            progress = prog
        
        delta_t = 10*(1 - (progress**5))
        concession = False
        ensmalled = 0
        # weight_copy = self._weight_est.copy()
        domain = self._profile.getProfile().getDomain()
        fla_change = True
        for issue in domain.getIssues():
            
            Fi1 = self._make_fi(issue, domain, self._window_2)
            
            # print(Fi1)
            Fi2 = self._make_fi(issue, domain, self._window_1)
            Fi1arr = []
            Fi2arr = []
            for k,v in Fi1.items():
                Fi1arr.append(v*1000)
                Fi2arr.append(Fi2.get(k)*1000)
            # print(Fi1arr)
            # print(Fi2arr)
            p_val = chisquare(Fi1arr, f_exp=Fi2arr,  axis= None)[1]
            
            # p_val = self._chi_squared_test(Fi1, Fi2)
            # print(p_val)
            if p_val>0.05:
                # print("same weights")
                unchanged.append(issue)
            else:
                fla_change = True
                # print("not same weights")
                e1 = self._estimate_calc(Fi1, issue)
                e2 = self._estimate_calc(Fi2, issue)
                # print(" %f %f"%(e1,e2))
                if e1 < e2:
                    ensmalled += (e2-e1)
                    print("CONCEDED! %f"%(e2-e1))
                    concession = True
                
        if concession and ((self._update_in_fairness and self._state>=4)or self._state<4):
            print("UPDATE VALUES")
            for i in unchanged:
                self._weight_est[i]+=delta_t
            sum = 0
            # print(self._weight_est)
            for k,v in self._weight_est.items():
                
                sum+=v
            if sum == 0:
                return ensmalled if concession else 0
            for k,v in self._weight_est.items():
                self._weight_est[k]=v/sum
        return ensmalled if concession else 0
    def _make_fi(self, issue: str, domain: Domain, window: "List[Bid]")->"dict[str,float]":
        values = domain.getIssuesValues().get(issue)
        n = values.size()
        out: dict[str, float]={}
        for i in range (n):
            value = values.get(i).getValue()
            for bid in window:
                count: float = 1
                if bid.getIssueValues().get(issue).getValue() == value:
                    count+=1
                count = (count)/(n+self._k)
                out[value] = count
        # print(out)
        return out
    def _estimate_calc(self, Fi: "dict[str,float]", issue: str)-> float:
        out: float = 0
        for k,v in Fi.items():
            out += self._total_est.get(issue).get(k)*v
        return out
    def _op_profile(self, b: Bid)->float:
        if self._opponent_profile is not None:
            return self._opponent_profile.getProfile().getUtility(b)
        out: Decimal = 0
        for k,v in b.getIssueValues().items():
            out+= Decimal(self._weight_est.get(k)*self._total_est.get(k).get(v.getValue()))
        return out
    # For the dynamic window creation
    def _make_windows(self):
        y_train = self._history.copy()
        y_train=list(map(lambda b: self._profile.getProfile().getUtility(b), y_train))
        x_train = self._x_train
        
        X_train = x_train[:, np.newaxis]
        model = make_pipeline(SplineTransformer(n_knots=int(floor(3*len(y_train)/4)), degree=15), Ridge(alpha=1e-3))
        model.fit(X_train, y_train)
        y_res = model.predict(X_train)
        new_arr = []
        for i in range(len(y_train)):
            new_arr.append(y_res[i])
        array = np.reshape(y_res, (-1,1))
        clustering = AgglomerativeClustering(n_clusters=ceil(2*len(array)/(self._k_max+self._k_min)), connectivity = self._connection_matrix, affinity='euclidean',
linkage='ward', distance_threshold=None,compute_full_tree=False).fit(array)
        windows = []
        curr_window = []
        prev_label=clustering.labels_[0]
        for i in range(len(self._history)):
            if clustering.labels_[i]!=prev_label:
                if len(curr_window)<self._k_min:
                    print("TOO SMALL :(")
                elif len(curr_window)>self._k_max:
                    while len(curr_window)>(2*self._k_max):
                        buff = curr_window[:self._k_max]
                        windows.append(buff)
                        curr_window = curr_window[self._k_max:]
                    while len(curr_window)>(self._k_min):
                        buff = curr_window[:int((self._k_min+self._k_max)/2)]
                        windows.append(buff)
                        curr_window = curr_window[int((self._k_min+self._k_max)/2):]
                else:
                    windows.append(curr_window)
                    curr_window=[]
            curr_window.append(self._history[i])
            prev_label=clustering.labels_[i]
        return windows
            
        
