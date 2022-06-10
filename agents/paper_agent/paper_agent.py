from decimal import Decimal
import logging

from random import randint
from typing import cast, List
from scipy.stats import chisquare

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
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
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressRounds import ProgressRounds



class Paper_Agent(DefaultParty):
    """
    Template agent that offers random bids until a bid with sufficient utility is offered.
    """

    def __init__(self):
        super().__init__()
        self._best_bids_so_far = []
        self._k = 15
        self._window_1 = []
        self._window_2 = []
        self._window_3 = []
        self._utarget = 0.9
        self._total_est: dict[str,dict[str, float]] = {}
        self._weight_est: dict[str,float] = {}
        self._init_flag = True
        self._history = []
        self._prefer_bits = []
        self._bids_set: set = {None}
        self._state = 0
        self._cmax = 0.09
        self._flag_window_update = False
        self.getReporter().log(logging.INFO, "party is initialized")
        self._profile = None
        self._flag = True
        self._last_received_bid: Bid = None

    def notifyChange(self, info: Inform):
        """This is the entry point of all interaction with your agent after is has been initialised.

        Args:
            info (Inform): Contains either a request for action or information.
        """

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
        # for k, v in self._total_est.items():
        #     print(k)
        #     for k1, v1 in v.items():
        #         print("   %s %f"%(k1, v1))
        # for k, v in self._weight_est.items():
        #     print("%s %f"%(k,v))
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
    def getDescription(self) -> Decimal:
        return "PAPER agent for Collaborative AI course"
    def _set_u_target(self)->bool:
        copy_s = self._utarget
        profile = self._profile.getProfile()
        ur = []
        for bid in self._window_2:
            ur.append(profile.getUtility(bid))
        ur.sort()
        ur = ur[:5]
        # print(ur)
        median = ur[2]
        

        ur2 = []
        for bid in self._window_1:
            ur2.append(profile.getUtility(bid))
        ur2.sort()
        ur2 = ur2[:5]
        # print(ur2)
        median2 = ur2[2]
        # print(list(map(lambda b: profile.getUtility(b),self._best_bids_so_far)))
        delta = Decimal.min(Decimal(self._cmax), median-median2)
        self._utarget=Decimal.min(Decimal(0.9),Decimal.max(profile.getUtility(self._best_bids_so_far[0]),Decimal(self._utarget)-delta))
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
        if self._last_received_bid is None:
            print("NO BIDS?")
        else:
            profile = self._profile.getProfile()
            self._history.append(self._last_received_bid)
            self._value_estimate()
            self._best_bids_so_far.append(self._last_received_bid)
            self._best_bids_so_far.sort(key = lambda b : profile.getUtility(b), reverse=True)
            self._best_bids_so_far = self._best_bids_so_far[:10]
                
            self._window_3.append(self._last_received_bid)
            if len( self._window_3)== self._k:
                self._window_1 = self._window_2
                self._window_2 = self._window_3
                self._window_3 = []
                if len(self._window_1) == self._k:
                    # print("ARRAYS: %d %d"%(len(self._window_1), len(self._window_2)))
                    out = self._set_u_target()
                    self._flag_window_update = True
                    out2 = self._weight_estimator()
                    if out2 !=0:
                        self._state +=1
                    # print("two us %f %f"%(out, self._utarget))
                    if self._utarget>=out:
                        self._utarget=Decimal(out)-Decimal(out2)/Decimal(len(self._profile.getProfile().getDomain().getIssues()))
                        # print("UTIL TARGET NEW %f"%self._utarget)
                    if self._state == 0:
                        self._state = 1
        if self._isGood(self._last_received_bid):
            # if so, accept the offerr
            print("ACCEPTING %d"%self._progress.get(0))
            action = Accept(self._me, self._last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            bid = self._findBid()
            action = Offer(self._me, bid)

        # send the action
        self.getConnection().send(action)
    def _value_estimate(self):
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

    def _weight_estimator(self)->int:
        unchanged = []
        # profile = self._profile.getProfile()
        progress = self._progress.get(0)
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
                    # print("CONCEDED! %f"%(e2-e1))
                    concession = True
                
        if concession:
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
    def _chi_squared_test(self, A: "dict[str,float]", B: "dict[str, float]")->float:
        result: float = 0
        for k,v in B.items():
            result += ((A.get(k)-v)**2)/v
        return result
        
    def _df_dep(self, df: int):
        out = [3.841,   5.991,   7.815, 9.488,11.070, 12.592,  14.067,  15.507,  16.919,  18.307, 19.675,  21.026,  22.362,   23.685, 24.996,  26.296,  27.587, 28.869,  30.144,  31.410,  32.671,  33.924,  35.172, 36.415,  37.652,  38.885,  40.113,  41.337,  42.557,  43.773,  44.985,  46.194,  47.400,  48.602 , 49.802,   50.998,  52.192,  53.384,  54.572 ,   55.758, 56.942, 58.124,  59.304,   60.481,   61.656,   62.830,   64.001,  65.171, 66.339,67.505, 68.669,  69.832,  70.993,  72.153,  73.311,  74.468,   75.624,  76.778,   77.931,  79.082,  80.232,  81.381,  82.529,  83.675,  84.821,   85.965,  87.108,   88.250,   89.391,   90.531,  91.670, 92.808,  93.945,  95.081,  96.217,  97.351,  98.484,  99.617,  100.749, 101.879,  103.010, 104.139,  105.267, 106.395, 107.522, 108.648,  109.773, 110.898,  112.022, 113.145,  114.268,  115.390,  116.511,  117.632,   118.752,   119.871,  120.990,  122.108, 123.225,  124.342]
        return out[df-2]
    # method that checks if we would agree with an offer
    def _isGood(self, bid: Bid) -> bool:
        if bid is None:
            return False
        profile = self._profile.getProfile()

        # progress = self._progress.get(0)

        # very basic approach that accepts if the offer is valued above 0.6 and
        # 80% of the rounds towards the deadline have passed
        return profile.getUtility(bid) > self._utarget
    def _rand_Bid(self) -> Bid:
        domain = self._profile.getProfile().getDomain()
        all_bids = AllBidsList(domain)
        
        while True:
            bid = all_bids.get(randint(0, all_bids.size() - 1))
            if self._isGood(bid):
                if bid in self._bids_set:
                    if randint(0,3) == 1:
                        break
                else:
                    self._bids_set.add(bid)
                    break
        # print("BID %f "%self._profile.getProfile().getUtility(bid))
        return bid
    def _findBid(self) -> Bid:
        # compose a list of all possible bids
        profile = self._profile.getProfile()
        if self._progress.get(0) > 0.98:
            # print("PROGRESS")
            if len(self._best_bids_so_far) > 0:
                bid = self._best_bids_so_far.pop()
                # if len(self._best_bids_so_far)>1:
                #     self._best_bids_so_far = self._best_bids_so_far[1:]
                # else:
                #     self._best_bids_so_far = []
                if profile.getUtility(bid) > 0.5:
                    return bid
                if len(self._window_1) == self._k:
                    # print("ARRAYS: %d %d"%(len(self._window_1), len(self._window_2)))
                    self._set_u_target()
        if self._isGood(self._last_received_bid):
            # if so, accept the offerr
            print("ACCEPTING %d"%self._progress.get(0))
            action = Accept(self._me, self._last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            bid = self._findBid()
            action = Offer(self._me, bid)

        # send the action
        self.getConnection().send(action)

    # method that checks if we would agree with an offer
    def _isGood(self, bid: Bid) -> bool:
        if bid is None:
            return False
        profile = self._profile.getProfile()

        # progress = self._progress.get(0)

        # very basic approach that accepts if the offer is valued above 0.6 and
        # 80% of the rounds towards the deadline have passed
        return profile.getUtility(bid) > self._utarget
    def _rand_Bid(self) -> Bid:
        domain = self._profile.getProfile().getDomain()
        all_bids = AllBidsList(domain)
        
        while True:
            bid = all_bids.get(randint(0, all_bids.size() - 1))
            if self._isGood(bid):
                if bid in self._bids_set:
                    if randint(0,3) == 1:
                        break
                else:
                    self._bids_set.add(bid)
                    break
        # print("BID %f "%self._profile.getProfile().getUtility(bid))
        return bid
    def _op_profile(self, b: Bid)->float:
        out: float = 0
        for k,v in b.getIssueValues().items():
            out+= (self._weight_est.get(k)*self._total_est.get(k).get(v.getValue()))
        return out
    def _findBid(self) -> Bid:
        # print(self._state)
        # compose a list of all possible bids
        profile = self._profile.getProfile()
        if self._progress.get(0) > 0.99:
            print("PROGRESS")
            if len(self._best_bids_so_far) > 0: 
                bid = self._best_bids_so_far.pop()
                if profile.getUtility(bid) > 0.5:
                    return bid 
                else:
                    return self._rand_Bid()
        if self._state >= 1 and self._state<4:
            self._prefer_bits = []
            return self._rand_Bid()
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
            out = self._prefer_bits.pop()
            return out
        else:
            # print("Third strat")
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
                    self._prefer_bits.sort(key = lambda b : profile.getUtility(b)+profile.getUtility(b)*Decimal(self._op_profile(b)), reverse=False)
                    if len(self._prefer_bits)>self._k*4:
                        self._prefer_bits = self._prefer_bits[1:]
                    # print( profile.getUtility(self._prefer_bits[0]))
            out = self._prefer_bits.pop()
            return out


            
        
            
        

        
