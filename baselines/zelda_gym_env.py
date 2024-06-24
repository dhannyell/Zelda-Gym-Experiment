import json
from math import floor
from pathlib import Path
from random import choice
import uuid

from einops import rearrange
from gymnasium import Env, spaces
from matplotlib import pyplot as plt
import numpy as np
import hnswlib

import pandas as pd
from pyboy.utils import WindowEvent
from pyboy.pyboy import PyBoy
import mediapy as media

from memory_addresses import *

class ZeldaGymEnv(Env):
    def __init__(self,config) -> None:
        self.debug = config['debug']
        self.s_path = config['session_path']
        self.save_final_state = config['save_final_state']
        self.print_rewards = config['print_rewards']
        self.vec_dim = 4320 #1000
        self.headless = config['headless']
        self.num_elements = 20000 # max
        self.init_state = config['init_state']
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        self.early_stopping = config['early_stop']
        self.save_video = config['save_video']
        self.fast_video = config['fast_video']
        self.video_interval = 256 * self.act_freq
        self.downsample_factor = 2
        self.frame_stacks = 3
        self.explore_weight = 1 if 'explore_weight' not in config else config['explore_weight']
        self.use_screen_explore = True if 'use_screen_explore' not in config else config['use_screen_explore']
        self.similar_frame_dist = config['sim_frame_dist']
        self.reward_scale = 1 if 'reward_scale' not in config else config['reward_scale']
        self.extra_buttons = False if 'extra_buttons' not in config else config['extra_buttons']
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self.s_path.mkdir(exist_ok=True)
        self.reset_count = 0
        self.steps_to_wait = 0
        self.all_runs = []

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B
        ]
        
        if self.extra_buttons:
            self.valid_actions.extend([
                WindowEvent.PRESS_BUTTON_START,
                WindowEvent.PASS
            ])

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        self.output_shape = (36, 40, 3)
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16
        self.output_full = (
            self.output_shape[0] * self.frame_stacks + 2 * (self.mem_padding + self.memory_height),
                            self.output_shape[1],
                            self.output_shape[2]
        )

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8)

        head = 'null' if config['headless'] else 'SDL2'

        self.pyboy = PyBoy(
                gamerom=config['gb_path'],
                window=head,
                debug=self.debug
            )

        if not config['headless']:
            self.pyboy.set_emulation_speed(6)

        self.reset()

    def reset(self, seed=None, options=None):
        self.seed = seed
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)
        
        self.init_map_mem()

        self.recent_memory = np.zeros((self.output_shape[1]*self.memory_height, 3), dtype=np.uint8)
        
        self.recent_frames = np.zeros(
            (self.frame_stacks, self.output_shape[0], 
             self.output_shape[1], self.output_shape[2]),
            dtype=np.uint8)

        self.agent_stats = []
        
        if self.save_video:
            base_dir = self.s_path / Path('rollouts')
            base_dir.mkdir(exist_ok=True)
            full_name = Path(f'full_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            model_name = Path(f'model_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.full_frame_writer.__enter__()
            self.model_frame_writer = media.VideoWriter(base_dir / model_name, self.output_full[:2], fps=60)
            self.model_frame_writer.__enter__()
       
        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 3
        self.total_healing_rew = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0
        self.current_map = 0
        self.rupees = 0
        self.enemies_killed = 0
        self.explore_reward = 0
        self.owl_talks = 0
        self.entered_dungeons = {}
        self.aux = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        self.key_pressed = 0

        return self.render(), {}

    def render(self, reduce_res=True, add_memory=True, update_mem=True):     
        game_pixels_render = self.pyboy.screen.ndarray # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (255*np.resize(game_pixels_render, self.output_shape)).astype(np.uint8)
            if update_mem:
                self.recent_frames[0] = game_pixels_render
            if add_memory:
                pad = np.zeros(
                    shape=(self.mem_padding, self.output_shape[1], 3), 
                    dtype=np.uint8)
                game_pixels_render = np.concatenate(
                    (
                        self.create_exploration_memory(), 
                        pad,
                        self.create_recent_memory(),
                        pad,
                        rearrange(self.recent_frames, 'f h w c -> (f h) w c')
                    ),
                    axis=0)
        return game_pixels_render
    
    def step(self, action):
        self.run_action_on_emulator(action)
        self.append_agent_stats(action)

        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        obs_memory = self.render()

        self.update_seen_coords()
            
        self.update_heal_reward()

        new_reward, new_prog = self.update_reward()
        
        self.last_health = self.read_hp()

        # shift over short term reward memory
        self.recent_memory = np.roll(self.recent_memory, 3)
        self.recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        self.recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        self.recent_memory[0, 2] = min(new_prog[2] * 128, 255)

        step_limit_reached = self.check_if_done()

        self.save_and_print_info(step_limit_reached, obs_memory)

        self.step_count += 1

        return obs_memory, new_reward*0.1, False, step_limit_reached, {}

    def check_if_done(self):
        if self.early_stopping:
            done = False
            if self.step_count > 128 and self.recent_memory.sum() < (255 * 1):
                done = True
        else:
            done = self.step_count >= self.max_steps
        
        return done
    
    def init_map_mem(self):
        self.seen_coords = {}

    def update_reward(self):
        # compute reward
        old_prog = self.group_rewards()
        self.progress_reward = self.get_game_state_reward()
        new_prog = self.group_rewards()
        new_total = sum([val for _, val in self.progress_reward.items()])
        new_step = new_total - self.total_reward
        if new_step < 0 and self.read_hp() > 0:
            self.save_screenshot('neg_reward')
    
        self.total_reward = new_total
        return (new_step, 
                   (new_prog[0]-old_prog[0], 
                    new_prog[1]-old_prog[1], 
                    new_prog[2]-old_prog[2])
               )
    
    def save_and_print_info(self, done, obs_memory):
        if self.print_rewards:
            prog_string = f'step: {self.step_count:6d}'
            for key, val in self.progress_reward.items():
                prog_string += f' {key}: {val:5.2f}'
            prog_string += f' sum: {self.total_reward:5.2f}'
            print(f'\r{prog_string}', end='', flush=True)
        
        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f'curframe_{self.instance_id}.jpeg'), 
                self.render(reduce_res=False))

        if self.print_rewards and done:
            print('', flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path('final_states')
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_small.jpeg'), 
                    obs_memory)
                plt.imsave(
                    fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg'), 
                    self.render(reduce_res=False))

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()

        if done:
            self.all_runs.append(self.progress_reward)
            with open(self.s_path / Path(f'all_runs_{self.instance_id}.json'), 'w') as f:
                json.dump(self.all_runs, f)
            pd.DataFrame(self.agent_stats).to_csv(
                self.s_path / Path(f'agent_stats_{self.instance_id}.csv.gz'), compression='gzip', mode='a')
            
        
    def append_agent_stats(self, action):
        x_pos = self.read_m(X_POS_ADDRESS)
        y_pos = self.read_m(Y_POS_ADDRESS)
        map_n = self.read_m(MAP_N_ADDRESS)
        levels = [self.read_m(a) for a in ITEMS_LEVEL_ADDRESSES]
        if self.use_screen_explore:
            expl = ('frames', self.get_explore_reward())
        else:
            expl = ('coord_count', len(self.seen_coords))
        self.agent_stats.append({
            'step': self.step_count,
            'map_location': self.get_map_location(map_n),
            'x': x_pos, 
            'y': y_pos, 
            'map': map_n,
            'last_action': action,
            'levels': levels, 
            'levels_sum': sum(levels),
            'inventory_items': self.get_inventory_items(),
            'hp': self.read_hp(),
            expl[0]: expl[1],
            'deaths': self.died_count,
            'enemies_killed': self.enemies_killed,
            'healr': self.total_healing_rew,
            'event': self.progress_reward['event']
        })

    def get_inventory_items(self):
        inventory_items = self.read_m(INVENTORY_ITEMS)

        events_dict = {i:inventory_items.count(i) for i in inventory_items if i > 0}

        return len(events_dict)

    def get_rumber_of_rupees(self):
        rupees = [self.read_m(a) for a in NUMBER_OF_RUPEES]
        return (rupees[0] + rupees[1])/100

    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm

        state_scores = {
            'inventory_items': self.reward_scale*self.get_inventory_items()*2, 
            'heal': self.reward_scale*0.1*self.total_healing_rew,
            'dead': self.reward_scale*-0.1*self.died_count,
            'enemies_killed': self.reward_scale*self.get_number_of_killed_enimies()*0.5,
            'event': self.reward_scale*self.get_all_events()*2,
            'explore': self.explore_weight*self.get_explore_reward()
        }
        
        return state_scores
    
    def get_all_events(self):
        number_of_small_keys = self.read_m(NUMBER_OF_SMALL_KEYS)

        heart_pieces = self.read_m(NUMBER_OF_HEART_PIECES)
        world_status = self.read_m(WORLD_STATUS)

        events_dict = {i:world_status.count(i) for i in world_status}

        get_sword_event = events_dict.get(176,0)

        current_owl_talks = events_dict.get(160,0)

        tail_key_event = events_dict.get(240,0)

        entered_dungeon = self.read_m(DUNGEON_ADDRESS)
                                                                                          
        if entered_dungeon > 0 and entered_dungeon != 17:
            is_dungeon_alread_visited = self.entered_dungeons.get(entered_dungeon, 0)
            if is_dungeon_alread_visited == 0:
                self.entered_dungeons[entered_dungeon] = entered_dungeon


        self.owl_talks = current_owl_talks

        if get_sword_event > 0:
            self.owl_talks = current_owl_talks + 1

        events = [
            number_of_small_keys, 
            heart_pieces, 
            get_sword_event*2,
            self.owl_talks,
            tail_key_event,
            len(self.entered_dungeons)
        ]
        
        self.max_event_rew = sum(events)
        return self.max_event_rew

    def get_number_of_killed_enimies(self):
        old_enemies_killed = self.enemies_killed
        current_enemies_killed = self.read_m(NUMBER_ENEMIES_KILLED)

        if old_enemies_killed > current_enemies_killed:
           if self.aux < current_enemies_killed:
            self.enemies_killed = old_enemies_killed + 1
           self.aux = current_enemies_killed
        else:
            self.enemies_killed = current_enemies_killed

        return self.enemies_killed
    
    def get_dungeon_locations(self, dun_idx):
        dungeon_locations = {}                     

        if dun_idx in dungeon_locations.keys():
            return dungeon_locations[dun_idx]
        else:
            return "Unknown Dungeon"

    def get_map_location(self, map_idx):
        dungeon_number = self.read_m(DUNGEON_ADDRESS)
        if dungeon_number != 0:                                                             
            return self.get_dungeon_locations(dungeon_number)
        
        map_locations = {
            48:  "Weird Mr. Write",
            49:  "Telephone Booth",
            50:  "Goponga Swamp",
            64:  "Mysterious Woods",
            65:  "Mysterious Woods",
            66:  "Mysterious Woods",
            67:  "Mysterious Woods",
            68:  "Koholint Prairie",
            69:  "Crazy Tracy's Health Spa",
            70:  "Tabahl Wasteland",
            71:  "Tabahl Wasteland",
            80:  "Mysterious Woods",
            81:  "Mysterious Woods",
            82:  "Mysterious Woods",
            83:  "Mysterious Woods",
            84:  "Koholint Prairie",
            85:  "Koholint Prairie",
            86:  "Tabahl Wasteland",
            87:  "Tabahl Wasteland",
            96:  "Mysterious Woods",
            97:  "Mysterious Woods",
            98:  "Mysterious Woods",
            99:  "Mysterious Woods",
            100: "Koholint Prairie",
            101: "Witch's Hut",
            102: "Cemetery",
            103: "Cemetery",
            112: "Mysterious Woods",
            113: "Mysterious Woods",
            114: "Mysterious Woods",
            115: "Mysterious Woods",
            116: "Koholint Prairie",
            117: "Koholint Prairie",
            118: "Cemetery",
            119: "Cemetery",
            128: "Mysterious Woods",
            129: "Fishing Pond",
            130: "Quadruplet's House",
            131: "Dream Shrine",
            132: "Ukuku Prairie",
            144: "Mysterious Woods",
            145: "Mysterious Woods",
            160: "Mabe Village",
            161: "Madame MeowMeow's House",
            162: "Marin and Tarin's House",
            163: "Mabe Village",
            164: "Telephone Booth",
            176: "Village Library",
            177: "Old Man Ulrira's House",
            178: "Telephone Booth",
            179: "Trendy Game",
            180: "Ukuku Prairie",
            181: "Level 3 - Key Cavern",
            182: "Ukuku Prairie",
            183: "Ukuku Prairie",
            192: "South of the Village",
            193: "South of the Village",
            194: "South of the Village",
            195: "South of the Village",
            196: "Signpost Maze",
            208: "South of the Village",
            209: "South of the Village",
            210: "South of the Village",
            211: "Level 1 - Tail Cave",
            212: "Signpost Maze",
            213: "Signpost Maze",
            214: "Richard's Villa",
            224: "Toronbo Shores",
            225: "Toronbo Shores",
            226: "Toronbo Shores",
            227: "Sale's House O' Bananas",
            228: "Toronbo Shores",
            229: "Toronbo Shores",
            230: "Martha's Bay",
            240: "Toronbo Shores",
            241: "Toronbo Shores",
            242: "Toronbo Shores",
            243: "Toronbo Shores",
            244: "Toronbo Shores",
            245: "Toronbo Shores",
        }

        if map_idx in map_locations.keys():
            return map_locations[map_idx]
        else:
            return "Unknown Location"

    def update_heal_reward(self):
        cur_health = self.read_hp()
        if cur_health > self.last_health:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                if heal_amount > 0.5:
                    print(f'healed: {heal_amount}')
                    self.save_screenshot('healing')
                self.total_healing_rew += heal_amount * 4
            else:
                self.died_count += 1

    def save_screenshot(self, name):
        ss_dir = self.s_path / Path('screenshots')
        ss_dir.mkdir(exist_ok=True)
        plt.imsave(
            ss_dir / Path(f'frame{self.instance_id}_r{self.total_reward:.4f}_{self.reset_count}_{name}.jpeg'), 
            self.render(reduce_res=False))
        
    def get_levels_sum(self):
        items_level = [self.read_m(a) for a in ITEMS_LEVEL_ADDRESSES]
        items_level[1] -= 1 # subtract starting shield level
        return sum(items_level) 
    
    def get_levels_reward(self):
        explore_thresh = 2
        scale_factor = 1
        level_sum = self.get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum-explore_thresh) / scale_factor + explore_thresh
        self.max_level_rew = max(self.max_level_rew, scaled)
        return self.max_level_rew
    
    def update_seen_coords(self):
        x_pos = self.read_m(X_POS_ADDRESS)
        y_pos = self.read_m(Y_POS_ADDRESS)
        map_n = self.read_m(MAP_N_ADDRESS)
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        if self.get_levels_sum() >= 22 and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = len(self.seen_coords)
            self.seen_coords = {}
        
        self.seen_coords[coord_string] = self.step_count

    def group_rewards(self):
        prog = self.progress_reward
        # these values are only used by memory
        return (prog['event'] * 100 / self.reward_scale, 
                self.read_hp()*2000, 
                prog['explore'] * 150 / (self.explore_weight * self.reward_scale))

    def get_explore_reward(self):
        return round(sum(self.read_m(WORLD_STATUS))/128)
    
    def run_action_on_emulator(self, action):
        if self.read_hp() <= 0:
            for i in range(3):
                self.pyboy.send_input(WindowEvent.PRESS_ARROW_UP)
                self.pyboy.send_input(WindowEvent.RELEASE_ARROW_UP)
            
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            self.pyboy.tick()
  
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if not self.save_video and self.headless:
            self.pyboy.tick(render=False)
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6:
                    # release button 
                    self.pyboy.send_input(self.release_button[action - 4])
                if self.valid_actions[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if self.save_video and not self.fast_video:
                self.add_video_frame()
            if i == self.act_freq-1:
                self.pyboy.tick(render=True)
            self.pyboy.tick()
        if self.save_video and self.fast_video:
            self.add_video_frame()

    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render(reduce_res=False, update_mem=False))
        self.model_frame_writer.add_image(self.render(reduce_res=True, update_mem=False))
        
    def create_recent_memory(self):
        return rearrange(
            self.recent_memory,
            '(w h) c -> h w c',
            h=self.memory_height
        )
    
    def read_m(self, addr):
        return self.pyboy.memory[addr]
    
    def read_hp(self):
        return self.read_m(HP_ADDRESS)/8

    def create_exploration_memory(self):
        w = self.output_shape[1]
        h = self.memory_height

        def make_reward_channel(r_val):
            col_steps = self.col_steps
            max_r_val = (w-1) * h * col_steps
            # truncate progress bar. if hitting this
            # you should scale down the reward in group_rewards!
            r_val = min(r_val, max_r_val)
            row = floor(r_val / (h * col_steps))
            memory = np.zeros(shape=(h, w), dtype=np.uint8)
            memory[:, :row] = 255
            row_covered = row * h * col_steps
            col = floor((r_val - row_covered) / col_steps)
            memory[:col, row] = 255
            col_covered = col * col_steps
            last_pixel = floor(r_val - row_covered - col_covered) 
            memory[col, row] = last_pixel * (255 // col_steps)
            return memory
        
        level, hp, explore = self.group_rewards()
        full_memory = np.stack((
            make_reward_channel(level),
            make_reward_channel(hp),
            make_reward_channel(explore)
        ), axis=-1)
        
        #if self.get_badges() > 0:
        #    full_memory[:, -1, :] = 255

        return full_memory

    