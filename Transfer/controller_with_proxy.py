# -*- coding: utf-8 -*-
"""Controller the real model with proxy and RL.


Units: cm, kg, s.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Jun 13 2021"


import sys
import pathlib
import os

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

import Sofa
import json
import numpy as np
from stable_baselines3 import SAC, PPO
from common.server_proxy import start_server, start_client, add_action, get_result, close_scene

class ControllerWithProxy(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root = kwargs["root"]
        self.scale = kwargs["scale"]
        self.dt = round(self.root.dt.value, 3)
        self.env_name = kwargs["env_name"]
        self.config = kwargs["config"]
        self.waitingtime = self.config["waitingtime"]
        self.time_register_sofagym =  kwargs["time_register_sofagym"]
        self.time_register_inverse =  kwargs["time_register_inverse"]
        self.nb_step_inverse =  kwargs["nb_step_inverse"]
        self.specific_actions =  kwargs["specific_actions"]

        self.max_bounds = np.array(kwargs["max_bounds"])
        self.min_bounds = np.array(kwargs["min_bounds"])
        self.old_action = np.array(kwargs["init_action"])
        self.factor_commande = kwargs["factor_commande"]
        self.Kp = kwargs["Kp"]


        if "translation" in kwargs:
            self.translation = np.array(kwargs["translation"])
        else:
            self.translation = np.array([0, 0, 0])

        self.config.update({"inverse_scene_path": kwargs["inverse_scene_path"]})
        start_server()

        self.idx_control, self.idx = 0, 0
        self.obs = np.array(start_client("sofagym", self.env_name)["observation"])
        start_client("inverse", self.config)

        save_rl_path = kwargs["save_rl_path"]
        if kwargs["name_algo"] == "SAC":
            self.model = SAC.load(save_rl_path)
        elif kwargs["name_algo"] == "PPO":
            self.model =PPO.load(save_rl_path)

        self.current_pos = None
        self.treshold = 0.01
        self.old_loss = np.inf
        self.control_step = 0
        self.max_control_step = 250

        self.num_sofa_action = 0
        self.incr_time = 1

        self.do_registration = False
        self.registration_time = []
        self.writers = kwargs["writers"]
        self.do_action = True

        self.cumulative_sofagym = 0
        self.current_time = 0

        self.use_cumulative_reward_sofagym = kwargs["use_cumulative_reward_sofagym"]
        self.effectors = kwargs["effectors"]
        self.goals = kwargs["goals"]

        self.list_time = []
        self.list_rewards_sofagym = []
        self.list_reward_real = []
        self.list_points_sofagym = []
        self.list_points_real = []

        self.name_save = kwargs["name_save"]
        os.makedirs(self.name_save, exist_ok = True)


    def _update_writers_times(self):
        for writer in self.writers:
            writer.time.value = self.registration_time

    def reinit(self):
        self.idx_control, self.idx = 0, 0
        close_scene(["sofagym", "inverse"])
        close_scene(["sofagym"])

        self.obs = np.array(start_client(self.env_name)["observation"])
        start_client("inverse", self.config)
        self.do_action = True

    def _one_step_sofagym(self):
        if self.specific_actions is not None:
            action = self.specific_actions[self.num_sofa_action]
        else:
            action, _ = self.model.predict(self.obs, deterministic = True)

        print("[SOFAGYM]  >> Choose action: ", action)
        add_action("sofagym", "step", action = action.tolist())
        results = get_result("sofagym", timeout=200)
        self.obs = np.array(results["observation"]) #to work with baseline
        last_reward = results["reward"]
        infos = results["infos"]["infos"]
        reward = results["infos"]["reward"]
        points = results["infos"]["points"]

        print("[SOFAGYM]  >>  Reward:", last_reward)
        self.num_sofa_action+=1
        return infos, reward, points

    def _update_inverse_goal(self, position):
        print("[INFO]  >> Update goal position.")
        p = np.array(position)
        p = self.root.moveGoal.getCorrectedPID(p, self.Kp)

        add_action("inverse", "change_goal", new_pos=p.tolist())
        self.root.moveGoal.update_goal(np.array(position))

    def _one_step_inverse(self):
        add_action("inverse", "step", dt = self.dt, scale = self.nb_step_inverse)
        results = get_result("inverse", timeout = 200)
        print("[INVERSE]  >>  Acuators's value:", results["infos"]["actuation"])
        return np.array(results["infos"]["actuation"]), results["infos"]["effectorsPos"], results["infos"]["points"]

    def _registration_inverse(self, readTime):
        print("[INVERSE]  >> Infos time: ReadTime: ", readTime, " ; incr_time:", self.incr_time, " ; time_register_inverse = ", self.time_register_inverse, " ; idx_control =", self.idx_control)
        if (self.incr_time-1)%self.time_register_inverse==0 and self.idx_control==self.scale-1:
            print("[INVERSE]  >> Registration.")
            close_scene(["inverse"])
            self.config.update({"readTime": readTime})
            start_client("inverse", self.config)

    def _registration_sofagym(self, infos):
        print("[SOFAGYM]  >> Infos time: incr_time = ", self.incr_time, " ; time_register_sofagym = ", self.time_register_sofagym, " ; idx_control =", self.idx_control)
        if (self.incr_time-1)%self.time_register_sofagym==0 and self.idx_control==self.scale-1:
            print("[SOFAGYM]  >> Registration position. Infos:", infos)
            add_action("sofagym", "setInfos", infos = infos)


    def onAnimateBeginEvent(self, event):
        if self.idx == 0:
            self.root.getReward.update()

        print("\n[INFO] >>  Check timer for registration:", round(self.root.time.value, 3), round((self.scale*self.dt)*self.incr_time, 3))
        if round(self.root.time.value, 3) == round((self.scale*self.dt)*self.incr_time, 3):
            self.registration_time.append(round(self.root.time.value, 3))
            self._update_writers_times()
            self.do_registration = True
            self.incr_time+=1


        if self.idx < self.waitingtime:
            print("[INFO]  >> Wainting Time. (", self.idx+1, "/", self.waitingtime, ")")
        elif not self.do_registration and self.do_action:
            print("[INFO]  >> Step ", self.idx+1, "(", self.idx_control+1, "/", self.scale, ") - Action number:", self.num_sofa_action)

            if self.idx_control == 0:
                print("[INFO]  >> New current infos.")
                self.current_infos, self.current_reward, self.current_points = self._one_step_sofagym()
            if self.idx_control < self.scale:
                print("[INFO]  >> Goal number:", self.idx_control)
                print("[SOFAGYM]  >> Reward sofagym:", self.current_reward[self.idx_control])
                self._update_inverse_goal([self.current_infos[self.idx_control]])



            actuation, self.effectorsPos_inverse, self.inverse_points = self._one_step_inverse()
            self.root.moveGoal.update_pos(np.array([self.effectorsPos_inverse]))

            action_rescaling = self.root.actuate.action_rescaling(
                self.current_points[self.idx_control], self.inverse_points)
            actuation = self.old_action + self.factor_commande*(action_rescaling*actuation - self.old_action)
            actuation = np.maximum(np.minimum(actuation, self.max_bounds), self.min_bounds)
            self.old_action = actuation
            self.root.actuate._setValue(actuation)

        self.idx+=1

    def onAnimateEndEvent(self, event):
        if self.do_registration:
            self.root.actuate.save_actuators_state()
            self._registration_inverse(round(self.root.time.value-self.dt, 3))

            infos = self.root.actuate.getInfos()
            self._registration_sofagym(infos)
            self.do_registration = False
            self.idx_control = 0
            self.control_step = 0
        else:
            self.idx_control = (self.idx_control +1)%self.scale
            reward, done = self.root.getReward.getReward()
            if self.use_cumulative_reward_sofagym:
                self.cumulative_sofagym+= self.current_reward[self.idx_control]
                reward_sofagym = self.cumulative_sofagym
            else:
                reward_sofagym = self.current_reward[self.idx_control]
            self.do_action = not done
            print("[INFO]  >> Reward in the real robot:", reward)
            print("[INFO]  >> Reward in cumulative sofagym:", reward_sofagym)

            self.current_time+= 0.01

            self.list_time.append(self.current_time)
            self.list_rewards_sofagym.append(reward_sofagym)
            self.list_reward_real.append(reward)

            pos_effectors = self.effectors.MechanicalObject.position.value.tolist()
            pos_goals = self.goals.MechanicalObject.position.value.tolist()
            self.list_points_sofagym.append(pos_goals)
            self.list_points_real.append(pos_effectors)


            with open(self.name_save+"time.txt", 'w') as f:
                json.dump(self.list_time, f)
            with open(self.name_save+"rewards_sofagym.txt", 'w') as f:
                json.dump(self.list_rewards_sofagym, f)
            with open(self.name_save+"reward_real.txt", 'w') as f:
                json.dump(self.list_reward_real, f)
            with open(self.name_save+"points_sofagym.txt", 'w') as f:
                json.dump(self.list_points_sofagym, f)
            with open(self.name_save+"points_real.txt", 'w') as f:
                json.dump(self.list_points_real, f)










class ControllerInverseWithProxy(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root = kwargs["root"]
        self.scale = kwargs["scale"]
        self.dt = round(self.root.dt.value, 3)
        self.env_name = kwargs["env_name"]
        self.config = kwargs["config"]
        self.waitingtime = self.config["waitingtime"]
        self.time_register_sofagym =  kwargs["time_register_sofagym"]
        self.specific_actions =  kwargs["specific_actions"]

        if "translation" in kwargs:
            self.translation = np.array(kwargs["translation"])
        else:
            self.translation = np.array([0, 0, 0])


        start_server()

        self.idx_control, self.idx = 0, 0
        self.obs = np.array(start_client("sofagym", self.env_name)["observation"])

        save_rl_path = kwargs["save_rl_path"]
        if kwargs["name_algo"] == "SAC":
            self.model = SAC.load(save_rl_path)
        elif kwargs["name_algo"] == "PPO":
            self.model =PPO.load(save_rl_path)

        self.current_pos = None
        self.treshold = 0.01
        self.old_loss = np.inf
        self.control_step = 0

        self.num_sofa_action = 0
        self.incr_time = 1

        self.do_registration = False
        self.registration_time = []
        self.do_action = True

        self.cumulative_sofagym = 0
        self.current_time = 0

        self.use_cumulative_reward_sofagym = kwargs["use_cumulative_reward_sofagym"]
        self.effectors = kwargs["effectors"]
        self.goals = kwargs["goals"]

        self.list_time = []
        self.list_rewards_sofagym = []
        self.list_reward_real = []
        self.list_points_sofagym = []
        self.list_points_real = []

        self.name_save = kwargs["name_save"]
        os.makedirs(self.name_save, exist_ok = True)

    def reinit(self):
        self.idx_control, self.idx = 0, 0
        close_scene(["sofagym"])

        self.obs = np.array(start_client(self.env_name)["observation"])
        self.do_action = True

    def _one_step_sofagym(self):
        if self.specific_actions is not None:
            action = self.specific_actions[self.num_sofa_action]
        else:
            action, _ = self.model.predict(self.obs, deterministic = True)

        print("[SOFAGYM]  >> Choose action: ", action)
        add_action("sofagym", "step", action = action.tolist())
        results = get_result("sofagym", timeout=200)
        self.obs = np.array(results["observation"]) #to work with baseline
        last_reward = results["reward"]
        infos = results["infos"]["infos"]
        reward = results["infos"]["reward"]
        points = results["infos"]["points"]

        print("[SOFAGYM]  >>  Reward:", last_reward)
        self.num_sofa_action+=1
        return infos, reward, points

    def _update_inverse_goal(self, position):
        print("[INFO]  >> Update goal position.")
        p = np.array(position)
        self.root.moveGoal.update_goal(p)


    def _registration_sofagym(self, infos):
        print("[SOFAGYM]  >> Infos time: incr_time = ", self.incr_time, " ; time_register_sofagym = ", self.time_register_sofagym, " ; idx_control =", self.idx_control)
        if (self.incr_time-1)%self.time_register_sofagym==0 and self.idx_control==self.scale-1:
            print("[SOFAGYM]  >> Registration position. Infos:", infos)
            add_action("sofagym", "setInfos", infos = infos)


    def onAnimateBeginEvent(self, event):
        if self.idx == 0:
            self.root.getReward.update()

        print("\n[INFO] >>  Check timer for registration:", round(self.root.time.value, 3), round((self.scale*self.dt)*self.incr_time, 3))
        if round(self.root.time.value, 3) == round((self.scale*self.dt)*self.incr_time, 3):
            self.registration_time.append(round(self.root.time.value, 3))
            self.do_registration = True
            self.incr_time+=1


        if self.idx < self.waitingtime:
            print("[INFO]  >> Wainting Time. (", self.idx+1, "/", self.waitingtime, ")")
        elif not self.do_registration and self.do_action:
            print("[INFO]  >> Step ", self.idx+1, "(", self.idx_control+1, "/", self.scale, ") - Action number:", self.num_sofa_action)

            if self.idx_control == 0:
                print("[INFO]  >> New current infos.")
                self.current_infos, self.current_reward, self.current_points = self._one_step_sofagym()
            if self.idx_control < self.scale:
                print("[INFO]  >> Goal number:", self.idx_control)
                print("[SOFAGYM]  >> Reward sofagym:", self.current_reward[self.idx_control])
                self._update_inverse_goal([self.current_infos[self.idx_control]])

        self.idx+=1

    def onAnimateEndEvent(self, event):
        if self.do_registration:
            infos = self.root.getInfos.getInfos()
            self._registration_sofagym(infos)
            self.do_registration = False

            self.idx_control = 0
            self.control_step = 0
        else:
            self.idx_control = (self.idx_control +1)%self.scale
            reward, done = self.root.getReward.getReward()
            if self.use_cumulative_reward_sofagym:
                self.cumulative_sofagym+= self.current_reward[self.idx_control]
                reward_sofagym = self.cumulative_sofagym
            else:
                reward_sofagym = self.current_reward[self.idx_control]
            self.do_action = not done
            print("[INFO]  >> Reward in the real robot:", reward)
            print("[INFO]  >> Reward in cumulative sofagym:", reward_sofagym)

            self.current_time+= 0.01

            self.list_time.append(self.current_time)
            self.list_rewards_sofagym.append(reward_sofagym)
            self.list_reward_real.append(reward)

            pos_effectors = self.effectors.MechanicalObject.position.value.tolist()
            pos_goals = self.goals.MechanicalObject.position.value.tolist()
            self.list_points_sofagym.append(pos_goals)
            self.list_points_real.append(pos_effectors)


            with open(self.name_save+"time.txt", 'w') as f:
                json.dump(self.list_time, f)
            with open(self.name_save+"rewards_sofagym.txt", 'w') as f:
                json.dump(self.list_rewards_sofagym, f)
            with open(self.name_save+"reward_real.txt", 'w') as f:
                json.dump(self.list_reward_real, f)
            with open(self.name_save+"points_sofagym.txt", 'w') as f:
                json.dump(self.list_points_sofagym, f)
            with open(self.name_save+"points_real.txt", 'w') as f:
                json.dump(self.list_points_real, f)
