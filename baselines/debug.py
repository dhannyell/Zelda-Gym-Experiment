from pyboy import PyBoy
from pyboy.utils import WindowEvent
from pprint import pprint
from random import choice

from memory_addresses import *

class Zelda():
    enemies_killed = 0
    died_count = 0
    total_healing_rew = 0
    last_health = 0
    pyboy = None
    reward_scale = 4

    def __init__(self, pyboy) -> None:
        self.pyboy = pyboy
    
    def read_m(self, address):
        return self.pyboy.memory[address]
    
    def read_hp(self):
        return self.read_m(HP_ADDRESS)/8
    
    def update_heal_reward(self):
        cur_health = self.read_hp()
        if cur_health > self.last_health:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                if heal_amount > 0.5:
                    print(f'healed: {heal_amount}')
                self.total_healing_rew += heal_amount * 4
            else:
                self.died_count += 1

    def get_explore_reward(self):
        return round(sum(self.read_m(WORLD_STATUS))/128)
    
    def get_rumber_of_rupees(self):
        rupees = [self.read_m(a) for a in NUMBER_OF_RUPEES]
        return (rupees[0] + rupees[1])/100
    
    def get_number_of_killed_enimies(self):
        current_enemies_killed = self.enemies_killed
        self.enemies_killed = self.read_m(NUMBER_ENEMIES_KILLED)

        if self.enemies_killed < current_enemies_killed and self.read_hp() > 0:
            self.enemies_killed += current_enemies_killed
            
        return self.enemies_killed

    def read_number_of_rupees(self):
        rupees = [self.read_m(a) for a in NUMBER_OF_RUPEES]
        return rupees[0] + rupees[1]
    
    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm

        state_scores = {
            'heal': self.reward_scale*0.1*self.total_healing_rew,
            'dead': self.reward_scale*-0.1*self.died_count,
            'rupees': self.reward_scale*self.get_rumber_of_rupees(),
            'enemies_killed': self.reward_scale*self.get_number_of_killed_enimies(),
            'event': self.get_all_events()*2,
            'explore': self.get_explore_reward()
        }
        
        return state_scores

    def get_all_events(self):
        number_of_small_keys = self.read_m(NUMBER_OF_SMALL_KEYS)

        heart_pieces = self.read_m(NUMBER_OF_HEART_PIECES)
        world_status = self.read_m(WORLD_STATUS)

        events_dict = {i:world_status.count(i) for i in world_status}

        events = [
            number_of_small_keys, 
            heart_pieces, 
            events_dict.get(176,0), # get sword
            events_dict.get(160,0) # owl talked
        ]
        
        self.max_event_rew = sum(events)
        return self.max_event_rew

if __name__ == '__main__':
    pyboy = PyBoy('../zelda_law.gb', debug=True)
    with open('../skip_intro_with_shield.state', "rb") as f:
        pyboy.load_state(f)
    pyboy.set_emulation_speed(6)
    zelda = pyboy.game_wrapper

    zelda_obj = Zelda(pyboy)
    while pyboy.tick():
        #zelda_obj.update_heal_reward()

        print(zelda_obj.get_game_state_reward())
        pass

    pyboy.stop()