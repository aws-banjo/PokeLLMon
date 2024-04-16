import json
import os
import random
import time
from typing import Dict, List, Optional, Union

import boto3

from poke_env.data.gen_data import GenData
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.move import Move
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.side_condition import SideCondition
from poke_env.player.player import BattleOrder, Player

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)


def calculate_move_type_damage_multipier(
    type_1, type_2, type_chart, constraint_type_list
):
    TYPE_list = "BUG,DARK,DRAGON,ELECTRIC,FAIRY,FIGHTING,FIRE,FLYING,GHOST,GRASS,GROUND,ICE,NORMAL,POISON,PSYCHIC,ROCK,STEEL,WATER".split(
        ","
    )

    move_type_damage_multiplier_list = []

    if type_2:
        for type in TYPE_list:
            move_type_damage_multiplier_list.append(
                type_chart[type_1][type] * type_chart[type_2][type]
            )
        move_type_damage_multiplier_dict = dict(
            zip(TYPE_list, move_type_damage_multiplier_list)
        )
    else:
        move_type_damage_multiplier_dict = type_chart[type_1]

    effective_type_list = []
    extreme_type_list = []
    resistant_type_list = []
    extreme_resistant_type_list = []
    immune_type_list = []
    for type, value in move_type_damage_multiplier_dict.items():
        if value == 2:
            effective_type_list.append(type)
        elif value == 4:
            extreme_type_list.append(type)
        elif value == 1 / 2:
            resistant_type_list.append(type)
        elif value == 1 / 4:
            extreme_resistant_type_list.append(type)
        elif value == 0:
            immune_type_list.append(type)
        else:  # value == 1
            continue

    if constraint_type_list:
        extreme_type_list = list(
            set(extreme_type_list).intersection(set(constraint_type_list))
        )
        effective_type_list = list(
            set(effective_type_list).intersection(set(constraint_type_list))
        )
        resistant_type_list = list(
            set(resistant_type_list).intersection(set(constraint_type_list))
        )
        extreme_resistant_type_list = list(
            set(extreme_resistant_type_list).intersection(set(constraint_type_list))
        )
        immune_type_list = list(
            set(immune_type_list).intersection(set(constraint_type_list))
        )

    return (
        list(map(lambda x: x.capitalize(), extreme_type_list)),
        list(map(lambda x: x.capitalize(), effective_type_list)),
        list(map(lambda x: x.capitalize(), resistant_type_list)),
        list(map(lambda x: x.capitalize(), extreme_resistant_type_list)),
        list(map(lambda x: x.capitalize(), immune_type_list)),
    )


def move_type_damage_wraper(pokemon, type_chart, constraint_type_list=None):

    type_1 = None
    type_2 = None
    if pokemon.type_1:
        type_1 = pokemon.type_1.name
        if pokemon.type_2:
            type_2 = pokemon.type_2.name

    move_type_damage_prompt = ""
    (
        extreme_effective_type_list,
        effective_type_list,
        resistant_type_list,
        extreme_resistant_type_list,
        immune_type_list,
    ) = calculate_move_type_damage_multipier(
        type_1, type_2, type_chart, constraint_type_list
    )

    move_type_damage_prompt = ""
    if extreme_effective_type_list:
        move_type_damage_prompt = (
            move_type_damage_prompt
            + " "
            + ", ".join(extreme_effective_type_list)
            + f"-type attack is extremely-effective (4x damage) to {pokemon.species}."
        )

    if effective_type_list:
        move_type_damage_prompt = (
            move_type_damage_prompt
            + " "
            + ", ".join(effective_type_list)
            + f"-type attack is super-effective (2x damage) to {pokemon.species}."
        )

    if resistant_type_list:
        move_type_damage_prompt = (
            move_type_damage_prompt
            + " "
            + ", ".join(resistant_type_list)
            + f"-type attack is ineffective (0.5x damage) to {pokemon.species}."
        )

    if extreme_resistant_type_list:
        move_type_damage_prompt = (
            move_type_damage_prompt
            + " "
            + ", ".join(extreme_resistant_type_list)
            + f"-type attack is highly ineffective (0.25x damage) to {pokemon.species}."
        )

    if immune_type_list:
        move_type_damage_prompt = (
            move_type_damage_prompt
            + " "
            + ", ".join(immune_type_list)
            + f"-type attack is zero effect (0x damage) to {pokemon.species}."
        )

    return move_type_damage_prompt


class LLMPlayer(Player):
    def __init__(
        self,
        battle_format,
        # api_key="",
        backend="gpt-4-1106-preview",
        temperature=0.8,
        prompt_algo="io",
        log_dir=None,
        team=None,
        save_replays=None,
        account_configuration=None,
        server_configuration=None,
    ):

        super().__init__(
            battle_format=battle_format,
            team=team,
            save_replays=save_replays,
            account_configuration=account_configuration,
            server_configuration=server_configuration,
        )

        self._reward_buffer: Dict[AbstractBattle, float] = {}
        self._battle_last_action: Dict[AbstractBattle, Dict] = {}
        self.last_action = ""
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.backend = backend
        self.temperature = temperature
        self.log_dir = log_dir
        # self.api_key = api_key
        self.prompt_algo = prompt_algo
        self.gen = GenData.from_format(battle_format)
        with open("./poke_env/data/static/moves/moves_effect.json", "r") as f:
            self.move_effect = json.load(f)
        with open("./poke_env/data/static/moves/gen8pokemon_move_dict.json", "r") as f:
            self.pokemon_move_dict = json.load(f)
        with open("./poke_env/data/static/abilities/ability_effect.json", "r") as f:
            self.ability_effect = json.load(f)
        with open(
            "./poke_env/data/static/abilities/gen8pokemon_ability_dict.json", "r"
        ) as f:
            self.pokemon_ability_dict = json.load(f)
        with open("./poke_env/data/static/items/item_effect.json", "r") as f:
            self.item_effect = json.load(f)
        with open("./poke_env/data/static/items/gen8pokemon_item_dict.json", "r") as f:
            self.pokemon_item_dict = json.load(f)

        self.last_plan = ""
        self.SPEED_TIER_COEFICIENT = 0.1
        self.HP_FRACTION_COEFICIENT = 0.4

    def bedrock(
        self,
        system_prompt,
        user_prompt,
        model,
        temperature=0.7,
        json_format=False,
        seed=None,
        stop=[],
        max_tokens=200,
    ) -> str:

        output = call_bedrock_model(model, system_prompt, user_prompt, bedrock_runtime)
        return output
        # client = OpenAI(api_key=self.api_key)
        # print(client)
        # if json_format:
        #     response = client.chat.completions.create(
        #         response_format={"type": "json_object"},
        #         model=model,
        #         messages=[
        #             {"role": "system", "content": system_prompt},
        #             {"role": "user", "content": user_prompt}
        #         ],
        #         temperature=temperature,
        #         stream=False,
        #         # seed=seed,
        #         stop=stop,
        #         max_tokens=max_tokens
        #     )
        # else:
        #     response = client.chat.completions.create(
        #         model=model,
        #         messages=[
        #             {"role": "system", "content": system_prompt},
        #             {"role": "user", "content": user_prompt}
        #         ],
        #         temperature=temperature,
        #         stream=False,
        #         # seed=seed,
        #         max_tokens=max_tokens,
        #         stop=stop
        #     )
        # outputs = response.choices[0].message.content
        # # # log completion tokens
        # self.completion_tokens += response.usage.completion_tokens
        # self.prompt_tokens += response.usage.prompt_tokens

        # return outputs

    def _estimate_matchup(self, mon: Pokemon, opponent: Pokemon):
        score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
        score -= max(
            [mon.damage_multiplier(t) for t in opponent.types if t is not None]
        )
        if mon.base_stats["spe"] > opponent.base_stats["spe"]:
            score += self.SPEED_TIER_COEFICIENT
        elif opponent.base_stats["spe"] > mon.base_stats["spe"]:
            score -= self.SPEED_TIER_COEFICIENT

        score += mon.current_hp_fraction * self.HP_FRACTION_COEFICIENT
        score -= opponent.current_hp_fraction * self.HP_FRACTION_COEFICIENT

        return score

    def _should_dynamax(self, battle: AbstractBattle):
        n_remaining_mons = len([m for m in battle.team.values() if m.fainted is False])
        if battle.can_dynamax and self._dynamax_disable is False:
            # Last full HP mon
            if (
                len([m for m in battle.team.values() if m.current_hp_fraction == 1])
                == 1
                and battle.active_pokemon.current_hp_fraction == 1
            ):
                return True
            # Matchup advantage and full hp on full hp
            if (
                self._estimate_matchup(
                    battle.active_pokemon, battle.opponent_active_pokemon
                )
                > 0
                and battle.active_pokemon.current_hp_fraction == 1
                and battle.opponent_active_pokemon.current_hp_fraction == 1
            ):
                return True
            if n_remaining_mons == 1:
                return True
        return False

    def state_translate(self, battle: AbstractBattle):

        n_turn = 5
        if "p1" in list(battle.team.keys())[0]:
            context_prompt = f"Historical turns:\n" + "\n".join(
                battle.battle_msg_history.split("[sep]")[-1 * (n_turn + 1) :]
            ).replace("p1a: ", "").replace("p2a:", "opposing").replace(
                "Player1", "You"
            ).replace(
                "Player2", "Opponent"
            )
        else:
            context_prompt = f"Historical turns:\n" + "\n".join(
                battle.battle_msg_history.split("[sep]")[-1 * (n_turn + 1) :]
            ).replace("p2a: ", "").replace("p1a:", "opposing").replace(
                "Player2", "You"
            ).replace(
                "Player1", "Opponent"
            )

        battle_prompt = context_prompt + " Current battle state:\n"

        # number of fainted pokemon
        opponent_fainted_num = 0
        for _, opponent_pokemon in battle.opponent_team.items():
            if opponent_pokemon.fainted:
                opponent_fainted_num += 1

        opponent_unfainted_num = 6 - opponent_fainted_num
        opponent_hp_fraction = round(
            battle.opponent_active_pokemon.current_hp
            / battle.opponent_active_pokemon.max_hp
            * 100
        )
        opponent_stats = battle.opponent_active_pokemon.calculate_stats()
        opponent_boosts = battle.opponent_active_pokemon._boosts
        active_stats = battle.active_pokemon.stats
        active_boosts = battle.active_pokemon._boosts
        opponent_status = battle.opponent_active_pokemon.status
        opponent_is_dynamax = battle.opponent_active_pokemon.is_dynamaxed

        # Type information
        opponent_type = ""

        opponent_type_list = []
        if battle.opponent_active_pokemon.type_1:
            type_1 = battle.opponent_active_pokemon.type_1.name
            opponent_type += type_1.capitalize()
            opponent_type_list.append(type_1)

            if battle.opponent_active_pokemon.type_2:
                type_2 = battle.opponent_active_pokemon.type_2.name
                opponent_type = opponent_type + " and " + type_2.capitalize()
                opponent_type_list.append(type_2)

        try:
            if battle.opponent_active_pokemon.ability:
                opponent_ability = battle.opponent_active_pokemon.ability
            elif (
                len(self.pokemon_ability_dict[battle.opponent_active_pokemon.species])
                == 1
            ):
                opponent_ability = self.pokemon_ability_dict[
                    battle.opponent_active_pokemon.species
                ][0]
            else:
                opponent_ability = ""
        except:
            opponent_ability = ""

        if opponent_ability:
            try:
                ability_name = self.ability_effect[opponent_ability]["name"]
                ability_effect = self.ability_effect[opponent_ability]["effect"]
                opponent_ability = f"{ability_name}({ability_effect})"
            except:
                pass

        opponent_prompt = (
            f"Opponent has {opponent_unfainted_num} pokemons left.\n"
            + f"Opposing pokemon:{battle.opponent_active_pokemon.species},Type:{opponent_type},HP:{opponent_hp_fraction}%,Is dynamax:{opponent_is_dynamax},"
            + (
                f"Status:{self.check_status(opponent_status)},"
                if self.check_status(opponent_status)
                else ""
            )
            + (
                f"Attack:{opponent_stats['atk']},"
                if opponent_boosts["atk"] == 0
                else f"Attack:{round(opponent_stats['atk'] * self.boost_multiplier('atk', opponent_boosts['atk']))}({opponent_boosts['atk']} stage boosted),"
            )
            + (
                f"Defense:{opponent_stats['def']},"
                if opponent_boosts["def"] == 0
                else f"Defense:{round(opponent_stats['def'] * self.boost_multiplier('def', opponent_boosts['def']))}({opponent_boosts['def']} stage boosted),"
            )
            + (
                f"Special attack:{opponent_stats['spa']},"
                if opponent_boosts["spa"] == 0
                else f"Special attack:{round(opponent_stats['spa'] * self.boost_multiplier('spa', opponent_boosts['spa']))}({opponent_boosts['spa']} stage boosted),"
            )
            + (
                f"Special defense:{opponent_stats['spd']},"
                if opponent_boosts["spd"] == 0
                else f"Special defense:{round(opponent_stats['spd'] * self.boost_multiplier('spd', opponent_boosts['spd']))}({opponent_boosts['spd']} stage boosted),"
            )
            + (
                f"Speed:{opponent_stats['spe']},"
                if opponent_boosts["spe"] == 0
                else f"Speed:{round(opponent_stats['spe'] * self.boost_multiplier('spe', opponent_boosts['spe']))}({opponent_boosts['spe']} stage boosted),"
            )
            + (f"Ability:{opponent_ability}" if opponent_ability else "")
        )
        opponent_speed = round(
            opponent_stats["spe"] * self.boost_multiplier("spe", opponent_boosts["spe"])
        )

        team_move_type = []
        for move in battle.available_moves:
            if move.base_power > 0:
                team_move_type.append(move.type.name)

        for pokemon in battle.available_switches:
            for move in pokemon.moves.values():
                if move.base_power > 0:
                    team_move_type.append(move.type.name)

        opponent_move_type_damage_prompt = move_type_damage_wraper(
            battle.opponent_active_pokemon, self.gen.type_chart, team_move_type
        )

        if opponent_move_type_damage_prompt:
            opponent_prompt = opponent_prompt + opponent_move_type_damage_prompt + "\n"

        # Opponent active pokemon move
        opponent_move_prompt = ""
        if battle.opponent_active_pokemon.moves:
            for move_id, opponent_move in battle.opponent_active_pokemon.moves.items():
                if opponent_move.base_power == 0:
                    continue  # only count attack move

                if opponent_move.category.name == "SPECIAL":
                    opponent_spa = opponent_stats["spa"] * self.boost_multiplier(
                        "spa", opponent_boosts["spa"]
                    )
                    active_spd = active_stats["spd"] * self.boost_multiplier(
                        "spd", active_boosts["spd"]
                    )
                    power = round(opponent_spa / active_spd * opponent_move.base_power)

                elif opponent_move.category.name == "PHYSICAL":
                    opponent_atk = opponent_stats["atk"] * self.boost_multiplier(
                        "atk", opponent_boosts["atk"]
                    )
                    active_def = active_stats["atk"] * self.boost_multiplier(
                        "atk", active_boosts["atk"]
                    )
                    power = round(opponent_atk / active_def * opponent_move.base_power)
                else:
                    power = 0

                opponent_move_prompt += f"[{opponent_move.id},{opponent_move.type.name.capitalize()},Power:{power}],"
                opponent_type_list.append(opponent_move.type.name)

        if opponent_move_prompt:
            opponent_prompt = (
                opponent_prompt
                + f"{battle.opponent_active_pokemon.species} used moves:"
                + opponent_move_prompt
            )

        possible_move_prompt = ""
        try:
            possible_move_list = list(
                self.pokemon_move_dict[battle.opponent_active_pokemon.species].values()
            )
            possible_move_list.sort(key=lambda x: x[3], reverse=True)
            for move in possible_move_list:
                if move[2] > 0:
                    possible_move_prompt = (
                        possible_move_prompt
                        + f"[{move[0]},{move[1].lower()},Power:{move[2]}],"
                    )
                    opponent_type_list.append(move[1].upper())
        except:
            possible_move_prompt = ""

        if possible_move_prompt:
            opponent_prompt = (
                opponent_prompt
                + f"{battle.opponent_active_pokemon.species}'s all the possible attacks:"
                + possible_move_prompt
            )

        opponent_side_condition_list = (
            []
        )  # I should add the description for the side condition. and the status.
        for side_condition in battle.opponent_side_conditions:
            opponent_side_condition_list.append(
                " ".join(side_condition.name.lower().split("_"))
            )

        opponent_side_condition = ",".join(opponent_side_condition_list)
        if opponent_side_condition:
            opponent_prompt = (
                opponent_prompt
                + "Opponent team's side condition: "
                + opponent_side_condition
            )

        opponent_prompt += "\n"

        # The active pokemon
        active_hp_fraction = round(
            battle.active_pokemon.current_hp / battle.active_pokemon.max_hp * 100
        )
        active_status = battle.active_pokemon.status

        active_type = ""
        if battle.active_pokemon.type_1:
            active_type += battle.active_pokemon.type_1.name.capitalize()
            if battle.active_pokemon.type_2:
                active_type = (
                    active_type
                    + " and "
                    + battle.active_pokemon.type_2.name.capitalize()
                )

        active_move_type_damage_prompt = move_type_damage_wraper(
            battle.active_pokemon, self.gen.type_chart, opponent_type_list
        )
        active_speed = round(
            active_stats["spe"] * self.boost_multiplier("spe", active_boosts["spe"])
        )

        try:
            active_ability = self.ability_effect[battle.active_pokemon.ability]["name"]
            ability_effect = self.ability_effect[battle.active_pokemon.ability][
                "effect"
            ]
        except:
            active_ability = battle.active_pokemon.ability
            ability_effect = ""

        # item
        if battle.active_pokemon.item:
            try:
                active_item = self.item_effect[battle.active_pokemon.item]["name"]
                item_effect = self.item_effect[battle.active_pokemon.item]["effect"]
                active_item = f"{active_item}({item_effect})"
            except:
                active_item = battle.active_pokemon.item
        else:
            active_item = ""

        active_pokemon_prompt = (
            f"Your current pokemon:{battle.active_pokemon.species},Type:{active_type},HP:{active_hp_fraction}%,"
            + (
                f"Status:{self.check_status(active_status)},"
                if self.check_status(active_status)
                else ""
            )
            + (
                f"Attack:{active_stats['atk']},"
                if active_boosts["atk"] == 0
                else f"Attack:{round(active_stats['atk']*self.boost_multiplier('atk', active_boosts['atk']))}({active_boosts['atk']} stage boosted),"
            )
            + (
                f"Defense:{active_stats['def']},"
                if active_boosts["def"] == 0
                else f"Defense:{round(active_stats['def']*self.boost_multiplier('def', active_boosts['def']))}({active_boosts['def']} stage boosted),"
            )
            + (
                f"Special attack:{active_stats['spa']},"
                if active_boosts["spa"] == 0
                else f"Special attack:{round(active_stats['spa']*self.boost_multiplier('spa', active_boosts['spa']))}({active_boosts['spa']} stage boosted),"
            )
            + (
                f"Special defense:{active_stats['spd']},"
                if active_boosts["spd"] == 0
                else f"Special defense:{round(active_stats['spd']*self.boost_multiplier('spd', active_boosts['spd']))}({active_boosts['spd']} stage boosted),"
            )
            + (
                f"Speed:{active_stats['spe']}"
                if active_boosts["spe"] == 0
                else f"Speed:{round(active_stats['spe']*self.boost_multiplier('spe', active_boosts['spe']))}({active_boosts['spe']} stage boosted),"
            )
            + (
                f"(slower than {battle.opponent_active_pokemon.species})."
                if active_speed < opponent_speed
                else f"(faster than {battle.opponent_active_pokemon.species})."
            )
            + (
                f"Ability:{active_ability}({ability_effect}),"
                if ability_effect
                else f"Ability:{active_ability},"
            )
            + (f"Item:{active_item}" if active_item else "")
        )

        if active_move_type_damage_prompt:
            active_pokemon_prompt = (
                active_pokemon_prompt + active_move_type_damage_prompt + "\n"
            )

        side_condition_list = []
        for side_condition in battle.side_conditions:

            side_condition_name = " ".join(side_condition.name.lower().split("_"))
            if side_condition == SideCondition.SPIKES:
                effect = (
                    " (cause damage to your pokémon when switch in except flying type)"
                )
            elif side_condition == SideCondition.STEALTH_ROCK:
                effect = " (cause rock-type damage to your pokémon when switch in)"
            elif side_condition == SideCondition.STICKY_WEB:
                effect = " (reduce the speed stat of your pokémon when switch in)"
            elif side_condition == SideCondition.TOXIC_SPIKES:
                effect = " (cause your pokémon toxic when switch in)"
            else:
                effect = ""

            side_condition_name = side_condition_name + effect
            side_condition_list.append(side_condition_name)

        side_condition_prompt = ",".join(side_condition_list)

        if side_condition_prompt:
            active_pokemon_prompt = (
                active_pokemon_prompt
                + "Your team's side condition: "
                + side_condition_prompt
                + "\n"
            )

        # Move
        move_prompt = f"Your {battle.active_pokemon.species} has {len(battle.available_moves)} moves:\n"
        for i, move in enumerate(battle.available_moves):
            try:
                effect = self.move_effect[move.id]
            except:
                effect = ""

            if move.category.name == "SPECIAL":
                active_spa = active_stats["spa"] * self.boost_multiplier(
                    "spa", active_boosts["spa"]
                )
                opponent_spd = opponent_stats["spd"] * self.boost_multiplier(
                    "spd", active_boosts["spd"]
                )
                power = round(active_spa / opponent_spd * move.base_power)
                move_category = ""
            elif move.category.name == "PHYSICAL":
                active_atk = active_stats["atk"] * self.boost_multiplier(
                    "atk", active_boosts["atk"]
                )
                opponent_def = opponent_stats["def"] * self.boost_multiplier(
                    "def", active_boosts["def"]
                )
                power = round(active_atk / opponent_def * move.base_power)
                move_category = ""
            else:
                move_category = move.category.name.capitalize()
                power = 0

            move_prompt += (
                f"Move:{move.id},Type:{move.type.name.capitalize()},"
                + (f"{move_category}-move," if move_category else "")
                + f"Power:{power},Acc:{round(move.accuracy * self.boost_multiplier('accuracy', active_boosts['accuracy'])*100)}%"
            )

            if effect:
                move_prompt += f",Effect:{effect}"
            # whether is effective to the target.
            move_type_damage_prompt = move_type_damage_wraper(
                battle.opponent_active_pokemon, self.gen.type_chart, [move.type.name]
            )
            if move_type_damage_prompt and move.base_power:
                move_prompt += f'({move_type_damage_prompt.split("is ")[-1][:-1]})\n'
            else:
                move_prompt += "\n"

        # Switch
        switch_prompt = f"You have {len(battle.available_switches)} pokemons:\n"

        for i, pokemon in enumerate(battle.available_switches):

            type = ""
            if pokemon.type_1:
                type_1 = pokemon.type_1.name
                type += type_1.capitalize()
                if pokemon.type_2:
                    type_2 = pokemon.type_2.name
                    type = type + " and " + type_2.capitalize()

            hp_fraction = round(pokemon.current_hp / pokemon.max_hp * 100)

            stats = pokemon.stats
            switch_move_prompt = f" Moves:"
            for _, move in pokemon.moves.items():
                if move.base_power == 0:
                    continue  # only output attack move
                move_type_damage_prompt = move_type_damage_wraper(
                    battle.opponent_active_pokemon,
                    self.gen.type_chart,
                    [move.type.name],
                )
                if "2x" in move_type_damage_prompt:
                    damage_multiplier = "2"
                elif "4x" in move_type_damage_prompt:
                    damage_multiplier = "4"
                elif "0.5x" in move_type_damage_prompt:
                    damage_multiplier = "0.5"
                elif "0.25x" in move_type_damage_prompt:
                    damage_multiplier = "0.25"
                elif "0x" in move_type_damage_prompt:
                    damage_multiplier = "0"
                else:
                    damage_multiplier = "1"

                switch_move_prompt += f"[{move.id},{move.type.name.capitalize()},{damage_multiplier}x damage],"

            if stats["spe"] < opponent_speed:
                speed_prompt = (
                    f"(slower than {battle.opponent_active_pokemon.species})."
                )
            else:
                speed_prompt = (
                    f"(faster than {battle.opponent_active_pokemon.species})."
                )

            switch_prompt += (
                f"Pokemon:{pokemon.species},Type:{type},HP:{hp_fraction}%,"
                + (
                    f"Status:{self.check_status(pokemon.status)}, "
                    if self.check_status(pokemon.status)
                    else ""
                )
                + f"Attack:{stats['atk']},Defense:{stats['def']},Special attack:{stats['spa']},Special defense:{stats['spd']},Speed:{stats['spe']}"
                + speed_prompt
                + switch_move_prompt
            )

            pokemon_move_type_damage_prompt = move_type_damage_wraper(
                pokemon, self.gen.type_chart, opponent_type_list
            )  # for defense

            if pokemon_move_type_damage_prompt:
                switch_prompt = switch_prompt + pokemon_move_type_damage_prompt + "\n"
            else:
                switch_prompt += "\n"

        if battle.active_pokemon.fainted:  # passive switching

            system_prompt = (
                f"You are a pokemon battler that targets to win the pokemon battle. Your {battle.active_pokemon.species} just fainted. Choose a suitable pokemon to continue the battle. Here are some tips:"
                " Compare the speeds of your pokemon to the opposing pokemon, which determines who take the move first."
                " Consider the defense state and type-resistance of your pokemon when its speed is lower than the opposing pokemon."
                " Consider the move-type advantage of your pokemon pokemon when its speed is higher than the opposing pokemon."
            )

            state_prompt = battle_prompt + opponent_prompt + switch_prompt

            return system_prompt, state_prompt

        else:  # take a move or active switch

            # system_prompt = (
            #     "You are a pokemon battler that targets to win the pokemon battle. You can choose to take a move or switch in another pokemon. Here are some battle tips:"
            #     " Use status-boosting moves like swordsdance, calmmind, dragondance, nastyplot strategically. The boosting will be reset when pokemon switch out."
            #     " Set traps like stickyweb, spikes, toxicspikes, stealthrock strategically."
            #     " When face to a opponent is boosting or has already boosted its attack/special attack/speed, knock it out as soon as possible, even sacrificing your pokemon."
            #     " if choose to switch, you forfeit to take a move this turn and the opposing pokemon will definitely move first. Therefore, you should pay attention to speed, type-resistance and defense of your switch-in pokemon to bear the damage from the opposing pokemon."
            #     " And If the switch-in pokemon has a slower speed then the opposing pokemon, the opposing pokemon will move twice continuously."
            # )

            system_prompt = f"""
You are a highly skilled and strategic Pokemon battler. Your primary goal is to make optimal move choices and switch decisions to defeat opposing Pokemon teams. Focus on knocking out the opposing Pokemon and only switch when absolutely necessary.

Your responses should have a confident, aggressive tone focused on maximizing damage output and securing KOs. Analyze the situation carefully, but prioritize attacking moves over switching whenever possible.

Given the current battle state with your active Pokemon, the opposing Pokemon, and any additional battlefield information, decide on the optimal action to take this turn - either choosing an attack move or switching to another Pokemon on your team if attacking is not viable.

Your decision should factor in:

    Type advantages/disadvantages
    Current boosts/debuffs on each Pokemon
    Entry hazards on the field
    Potential to set up for bigger damage later
    Revenge killing opportunities
    Preserving your own Pokemon's health, but not at the cost of missing KO opportunities

Use status-boosting moves like swords dance, calm mind, dragon dance, nasty plot strategically. The boosting will be reset when pokemon switch out. Set traps like stick yweb, spikes, toxic spikes, stealth rock strategically. When faced with a opponent that is boosting or has already boosted it's attack/special attack/speed, knock it out as soon as possible, even sacrificing your pokemon.    

If your active Pokemon has a reasonable chance to KO the opponent's Pokemon, even if it is low on health, prioritize attacking over switching. "Panic switching" will lead to poor outcomes and lost battles, so focus on attacking first and only switch when your active Pokemon is guaranteed to faint to the opponent's next move.

If you have just switched in a Pokemon, carefully consider the opponent's likely moves before switching again. Rapid switching gives your opponent free turns to set up or deal damage. Be confident in your switch-ins and aim to maintain offensive pressure.

Explain your reasoning step-by-step in arriving at your chosen action, emphasizing why attacking is the optimal play whenever possible and why you are confident in your choices.
<examples>
Example 1

Your Mesprit (full HP) vs Opponent's Metagross (7% HP)

Thinking process:

    Metagross outspeeds Mesprit and can hurt with with Meteor Mash
    Mesprit's Psychic attacks are not very effective against Metagross
    Attacking has a high chance to KO Metagross, whereas switching lets it get off a free hit
    Even at low HP, Mesprit's best play is to attack

Output move: Psychic
Example 2

Your Toxapex vs Opponent's Xurkitree

    Both at full health
    Toxapex has no recovery and is weak to Xurkitree's Electric attacks
    Toxapex is too slow to threaten Xurkitree and will get 2HKOed

Thinking process:

    Toxapex cannot win this matchup and needs to switch
    Potential switch-ins: Landorus-Therian, Garchomp, Seismitoad, etc.
    Switching is necessary to preserve Toxapex and bring in a counter
    The switch-in must be able to either tank Xurkitree's hits or threaten it with super-effective damage
    Landorus-Therian resists Electric, outspeeds, and can OHKO with Earthquake
    Confident that Landorus-Therian is the optimal switch-in to beat Xurkitree

Output: Switch to Landorus-Therian

Example 3
Your Ferrothorn (7 %) vs Opponent's Togakiss (70% HP)

Thinking process:
    Ferrothorn is very low at 7% HP and cannot survive another Air Slash from Togekiss, which is boosted by Nasty Plot. Rapid spin from Ferrothorn will do some damage to Togekiss but not enough to KO it. 

There is no point in trying to save Ferrothorn, so lets just attack here so we dont hurt are other pokemon

Output move: Rapid spin


Example 4
Your Duraludon (6 %) vs Opponent's Thundurus-Therian (100% HP)

Duraludon is at a very low 6% HP and cannot survive any attack from the opposing Thundurus-Therian. Flash Cannon and Body Press would be ineffective against Thundurus. Stealth Rock could be useful to set up entry hazards, but Duraludon likely won't survive to see the benefits.

Switching is a waste of time as Duraludon has already done enough, and i dont want to hurt another pokemon.

The optimal play is to stay in and just take the hit.

Output move: Flash Cannon
</examples>

For reference here was your last move: 
{self.last_action}

Remember, your goal is to win. Be decisive and go for KOs whenever possible. Switching should be a last resort, not a go-to option. If you do switch, choose a Pokemon that can threaten the opponent or tank their hits. Seize every opportunity to deal big damage and remove opposing threats from the field. Maintain offensive pressure and don't allow unnecessary free turns. 

If your previous move was a switch think long and hard before saying to switch again, explain why you will make two switches in a row which gives the opponent two free moves. Don't worry if the opponent Pokemon is strong because of boosts do not switch twice in a row, doing so will cause you to lose the match. Also, Do not worry about preserving pokemon that will not help in the battle any more, trying to preserve a pokemon that is about to faint will cost you the match. You play to win!!!
"""

            state_prompt = (
                battle_prompt
                + opponent_prompt
                + active_pokemon_prompt
                + move_prompt
                + switch_prompt
            )

            return system_prompt, state_prompt

    def parse(self, llm_output, battle):
        json_start = llm_output.find("{")
        json_end = llm_output.rfind("}") + 1  # find the first }
        json_content = llm_output[json_start:json_end]
        llm_action_json = json.loads(json_content)
        next_action = None
        if "move" in llm_action_json.keys():
            llm_move_id = llm_action_json["move"]
            llm_move_id = llm_move_id.replace(" ", "").replace("-", "")
            for i, move in enumerate(battle.available_moves):
                if move.id.lower() == llm_move_id.lower():
                    next_action = self.create_order(
                        move, dynamax=self._should_dynamax(battle)
                    )

        elif "switch" in llm_action_json.keys():
            llm_switch_species = llm_action_json["switch"]
            for i, pokemon in enumerate(battle.available_switches):
                if pokemon.species.lower() == llm_switch_species.lower():
                    next_action = self.create_order(pokemon)

        if next_action is None:
            raise ValueError("Value Error")
        return next_action

    def parse_new(self, llm_output, battle):
        json_start = llm_output.find("{")
        json_end = llm_output.rfind("}") + 1  # find the first }
        json_content = llm_output[json_start:json_end]
        llm_action_json = json.loads(json_content)
        next_action = None
        action = llm_action_json["decision"]["action"]
        target = llm_action_json["decision"]["target"]
        target = target.replace(" ", "").replace("_", "")
        if action.lower() == "move":
            for i, move in enumerate(battle.available_moves):
                if move.id.lower() == target.lower():
                    next_action = self.create_order(
                        move, dynamax=self._should_dynamax(battle)
                    )

        elif action.lower() == "switch":
            for i, pokemon in enumerate(battle.available_switches):
                if pokemon.species.lower() == target.lower():
                    next_action = self.create_order(pokemon)

        if next_action is None:
            raise ValueError("Value Error")

        return next_action

    def check_status(self, status):
        if status:
            if status.value == 1:
                return "burnt"
            elif status.value == 2:
                return "fainted"
            elif status.value == 3:
                return "frozen"
            elif status.value == 4:
                return "paralyzed"
            elif status.value == 5:
                return "poisoned"
            elif status.value == 7:
                return "toxic"
            elif status.value == 6:
                return "sleeping"
        else:
            return ""

    def boost_multiplier(self, state, level):
        if state == "accuracy":
            if level == 0:
                return 1.0
            if level == 1:
                return 1.33
            if level == 2:
                return 1.66
            if level == 3:
                return 2.0
            if level == 4:
                return 2.5
            if level == 5:
                return 2.66
            if level == 6:
                return 3.0
            if level == -1:
                return 0.75
            if level == -2:
                return 0.6
            if level == -3:
                return 0.5
            if level == -4:
                return 0.43
            if level == -5:
                return 0.36
            if level == -6:
                return 0.33
        else:
            if level == 0:
                return 1.0
            if level == 1:
                return 1.5
            if level == 2:
                return 2.0
            if level == 3:
                return 2.5
            if level == 4:
                return 3.0
            if level == 5:
                return 3.5
            if level == 6:
                return 4.0
            if level == -1:
                return 0.67
            if level == -2:
                return 0.5
            if level == -3:
                return 0.4
            if level == -4:
                return 0.33
            if level == -5:
                return 0.29
            if level == -6:
                return 0.25

    def choose_move(self, battle: AbstractBattle):

        # state_prompt = self.state_translate(battle)
        # return self.choose_random_move(battle)

        if battle.active_pokemon.fainted and len(battle.available_switches) == 1:
            next_action = BattleOrder(battle.available_switches[0])
            return next_action

        # state_prompt = self.state_translate(battle)
        system_prompt, state_prompt = self.state_translate(battle)  # add lower case

        if battle.active_pokemon.fainted:

            constraint_prompt_io = """Choose the most suitable pokemon to switch. Your output MUST be a JSON like: {"switch":"<switch_pokemon_name>"}\n"""
            constraint_prompt_cot = """Choose the most suitable pokemon to switch by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "switch":"<switch_pokemon_name>"}\n"""
            constraint_prompt_tot_1 = """Generate top-k (k<=3) best switch options. Your output MUST be a JSON like:{"option_1":{"action":"switch","target":"<switch_pokemon_name>"}, ..., "option_k":{"action":"switch","target":"<switch_pokemon_name>"}}\n"""
            constraint_prompt_tot_2 = """Select the best option from the following choices by considering their consequences: [OPTIONS]. Your output MUST be a JSON like:{"decision":{"action":"switch","target":"<switch_pokemon_name>"}}\n"""

        else:
            constraint_prompt_io = """Choose the best action and your output MUST be a JSON like: {"move":"<move_name>"} or {"switch":"<switch_pokemon_name>"}\n"""
            constraint_prompt_cot = """Choose the best action by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "move":"<move_name>"} or {"thought":"<step-by-step-thinking>", "switch":"<switch_pokemon_name>"}\n"""
            constraint_prompt_tot_1 = """Generate top-k (k<=3) best action options. Your output MUST be a JSON like: {"option_1":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>"}, ..., "option_k":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>"}}\n"""
            constraint_prompt_tot_2 = """Select the best action from the following choices by considering their consequences: [OPTIONS]. Your output MUST be a JSON like:"decision":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>"}\n"""

        remind_again = f"""
For reference here was your last move: 
{self.last_action}

Remember, your goal is to win. Be decisive and go for KOs whenever possible. Switching should be a last resort, not a go-to option. If you do switch, choose a Pokemon that can threaten the opponent or tank their hits. Seize every opportunity to deal big damage and remove opposing threats from the field. Maintain offensive pressure and don't allow unnecessary free turns. 

If your previous move was a switch think long and hard before saying to switch again, explain why you will make two switches in a row which gives the opponent two free moves.
"""

        state_prompt_io = state_prompt + constraint_prompt_io + remind_again
        state_prompt_cot = state_prompt + constraint_prompt_cot + remind_again
        state_prompt_tot_1 = state_prompt + constraint_prompt_tot_1 + remind_again
        state_prompt_tot_2 = state_prompt + constraint_prompt_tot_2 + remind_again

        print("===================")
        print(f"{self.backend} is making a move:")
        # print("\n")
        # print(state_prompt_io)
        # print("\n")

        if self.prompt_algo == "io":
            next_action = None
            for i in range(2):
                try:

                    start_time = time.time()
                    llm_output = self.bedrock(
                        system_prompt=system_prompt,
                        user_prompt=state_prompt_io,
                        model=self.backend,
                        temperature=self.temperature,
                        max_tokens=100,
                        # stop=["reason"],
                        json_format=True,
                    )
                    print(f"LLM call to {self.backend}: {time.time() - start_time}s")
                    print("LLM output:", llm_output)
                    next_action = self.parse(llm_output, battle)
                    print("Next action:", next_action)

                    with open(f"{self.log_dir}/output.jsonl", "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "turn": battle.turn,
                                    "system_prompt": system_prompt,
                                    "user_prompt": state_prompt_io,
                                    "llm_output": llm_output,
                                    "battle_tag": battle.battle_tag,
                                }
                            )
                            + "\n"
                        )
                    break
                except:
                    continue
            if next_action is None:
                next_action = self.choose_max_damage_move(battle)
            self.last_action = next_action
            return next_action

        # Self-consistency with k = 3
        elif self.prompt_algo == "sc":
            next_action1 = None
            next_action2 = None
            for i in range(2):
                try:
                    llm_output1 = self.bedrock(
                        system_prompt=system_prompt,
                        user_prompt=state_prompt_io,
                        model=self.backend,
                        temperature=self.temperature,
                        max_tokens=100,
                        json_format=True,
                    )
                    print("llm_output1:", llm_output1)
                    next_action1 = self.parse(llm_output1, battle)
                    break
                except:
                    continue

            for i in range(2):
                try:
                    llm_output2 = self.bedrock(
                        system_prompt=system_prompt,
                        user_prompt=state_prompt_io,
                        model=self.backend,
                        temperature=self.temperature,
                        max_tokens=100,
                        json_format=True,
                    )
                    print("llm_output2:", llm_output2)
                    next_action2 = self.parse(llm_output2, battle)
                    break
                except:
                    continue
            if next_action1 and next_action2:
                if next_action1.message == next_action2.message:
                    with open(f"{self.log_dir}/output.jsonl", "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "turn": battle.turn,
                                    "system_prompt": system_prompt,
                                    "user_prompt": state_prompt_io,
                                    "llm_output1": llm_output1,
                                    "llm_output2": llm_output2,
                                    "battle_tag": battle.battle_tag,
                                }
                            )
                            + "\n"
                        )
                    return next_action1
                else:
                    next_action3 = None
                    for i in range(2):
                        try:
                            llm_output3 = self.bedrock(
                                system_prompt=system_prompt,
                                user_prompt=state_prompt_io,
                                model=self.backend,
                                temperature=self.temperature,
                                max_tokens=100,
                                json_format=True,
                            )
                            print("llm_output3:", llm_output3)
                            next_action3 = self.parse(llm_output3, battle)
                            break
                        except:
                            continue
                    if next_action3:
                        with open(f"{self.log_dir}/output.jsonl", "a") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "turn": battle.turn,
                                        "system_prompt": system_prompt,
                                        "user_prompt": state_prompt_io,
                                        "llm_output1": llm_output1,
                                        "llm_output2": llm_output2,
                                        "llm_output3": llm_output3,
                                        "battle_tag": battle.battle_tag,
                                    }
                                )
                                + "\n"
                            )
                        return next_action3
                    else:
                        return next_action1
            next_action = self.choose_max_damage_move(battle)
            return next_action

        # Chain-of-thought
        elif self.prompt_algo == "cot":
            next_action = None
            for i in range(3):
                try:
                    llm_output = self.bedrock(
                        system_prompt=system_prompt,
                        user_prompt=state_prompt_cot,
                        model=self.backend,
                        temperature=self.temperature,
                        max_tokens=500,
                        # stop=["reason"],
                        json_format=True,
                    )
                    print("LLM output:", llm_output)
                    next_action = self.parse(llm_output, battle)
                    with open(f"{self.log_dir}/output.jsonl", "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "turn": battle.turn,
                                    "system_prompt": system_prompt,
                                    "user_prompt": state_prompt_cot,
                                    "llm_output": llm_output,
                                    "battle_tag": battle.battle_tag,
                                }
                            )
                            + "\n"
                        )
                    break
                except:
                    continue
            if next_action is None:
                next_action = self.choose_max_damage_move(battle)
            return next_action

        # Tree of thought, k = 3
        elif self.prompt_algo == "tot":
            llm_output1 = ""
            next_action = None
            for i in range(2):
                try:
                    llm_output1 = self.bedrock(
                        system_prompt=system_prompt,
                        user_prompt=state_prompt_tot_1,
                        model=self.backend,
                        temperature=self.temperature,
                        max_tokens=200,
                        json_format=True,
                    )
                    print("Phase 1 output:", llm_output1)
                    break
                except:
                    continue

            if llm_output1 == "":
                return self.choose_max_damage_move(battle)

            for i in range(2):
                try:
                    llm_output2 = self.bedrock(
                        system_prompt=system_prompt,
                        user_prompt=state_prompt_tot_2.replace(
                            "[OPTIONS]", llm_output1
                        ),
                        model=self.backend,
                        temperature=self.temperature,
                        max_tokens=100,
                        json_format=True,
                    )

                    print("Phase 2 output:", llm_output2)
                    next_action = self.parse_new(llm_output2, battle)
                    with open(f"{self.log_dir}/output.jsonl", "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "turn": battle.turn,
                                    "system_prompt": system_prompt,
                                    "user_prompt1": state_prompt_tot_1,
                                    "user_prompt2": state_prompt_tot_2,
                                    "llm_output1": llm_output1,
                                    "llm_output2": llm_output2,
                                    "battle_tag": battle.battle_tag,
                                }
                            )
                            + "\n"
                        )
                    break
                except:
                    continue

            if next_action is None:
                next_action = self.choose_max_damage_move(battle)
            return next_action

    def battle_summary(self):

        beat_list = []
        remain_list = []
        win_list = []
        tag_list = []
        for tag, battle in self.battles.items():
            beat_score = 0
            for mon in battle.opponent_team.values():
                beat_score += 1 - mon.current_hp_fraction

            beat_list.append(beat_score)

            remain_score = 0
            for mon in battle.team.values():
                remain_score += mon.current_hp_fraction

            remain_list.append(remain_score)
            if battle.won:
                win_list.append(1)

            tag_list.append(tag)

        return beat_list, remain_list, win_list, tag_list

    def reward_computing_helper(
        self,
        battle: AbstractBattle,
        *,
        fainted_value: float = 0.0,
        hp_value: float = 0.0,
        number_of_pokemons: int = 6,
        starting_value: float = 0.0,
        status_value: float = 0.0,
        victory_value: float = 1.0,
    ) -> float:
        """A helper function to compute rewards."""

        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = starting_value
        current_value = 0

        for mon in battle.team.values():
            current_value += mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value -= fainted_value
            elif mon.status is not None:
                current_value -= status_value

        current_value += (number_of_pokemons - len(battle.team)) * hp_value

        for mon in battle.opponent_team.values():
            current_value -= mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value += fainted_value
            elif mon.status is not None:
                current_value += status_value

        current_value -= (number_of_pokemons - len(battle.opponent_team)) * hp_value

        if battle.won:
            current_value += victory_value
        elif battle.lost:
            current_value -= victory_value

        to_return = (
            current_value - self._reward_buffer[battle]
        )  # the return value is the delta
        self._reward_buffer[battle] = current_value

        return to_return

    def choose_max_damage_move(self, battle: AbstractBattle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        return self.choose_random_move(battle)


# Call Mistral model
def call_mistral_large(system_prompt, prompt, bedrock_runtime):
    prompt_config = {
        "prompt": f"<s>[INST]{system_prompt} {prompt}[/INST]",
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.8,
    }

    body = json.dumps(prompt_config)

    modelId = "mistral.mistral-large-2402-v1:0"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("outputs")[0].get("text")
    return results


def call_mistral_8x7b(system_prompt, prompt, bedrock_runtime):
    prompt_config = {
        "prompt": system_prompt + prompt,
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.8,
    }

    body = json.dumps(prompt_config)

    modelId = "mistral.mixtral-8x7b-instruct-v0:1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("outputs")[0].get("text")
    return results


# Call Mistral model
def call_mistral_7b(system_prompt, prompt, bedrock_runtime):
    prompt_config = {
        "prompt": system_prompt + prompt,
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.8,
    }

    body = json.dumps(prompt_config)

    modelId = "mistral.mistral-7b-instruct-v0:2"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("outputs")[0].get("text")
    return results


# Call AI21 labs model
def call_ai21_ultra(system_prompt, prompt, bedrock_runtime):
    prompt_config = {
        "prompt": system_prompt + prompt,
        "maxTokens": 5147,
        "temperature": 0.7,
        "stopSequences": [],
    }

    body = json.dumps(prompt_config)

    modelId = "ai21.j2-ultra-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("completions")[0].get("data").get("text")
    return results


def call_ai21_mid(system_prompt, prompt, bedrock_runtime):
    prompt_config = {
        "prompt": system_prompt + prompt,
        "maxTokens": 5147,
        "temperature": 0.7,
        "stopSequences": [],
    }

    body = json.dumps(prompt_config)

    modelId = "ai21.j2-mid-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("completions")[0].get("data").get("text")
    return results


def claude_2_prompt_format(prompt: str) -> str:
    # Add headers to start and end of prompt
    return "\n\nHuman: " + prompt + "\n\nAssistant:"


# Call Claude model
def call_claude_3_opus(system_prompt, prompt, bedrock_runtime):

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-3-opus-20240229-v1:0"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("content")[0].get("text")
    return results


def call_claude_3_sonnet(system_prompt, prompt, bedrock_runtime):

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("content")[0].get("text")
    return results


def call_claude_3_haiku(system_prompt, prompt, bedrock_runtime):

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-3-haiku-20240307-v1:0"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("content")[0].get("text")
    return results


# Call Claude model
def call_claude_2_1(system_prompt, prompt, bedrock_runtime):
    prompt_config = {
        "prompt": claude_2_prompt_format(system_prompt + prompt),
        "max_tokens_to_sample": 4096,
        "temperature": 0.7,
        "top_k": 250,
        "top_p": 0.5,
        "stop_sequences": [],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-v2:1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("completion")
    return results


# Call Claude model
def call_claude_2(system_prompt, prompt, bedrock_runtime):
    prompt_config = {
        "prompt": claude_2_prompt_format(system_prompt + prompt),
        "max_tokens_to_sample": 4096,
        "temperature": 0.7,
        "top_k": 250,
        "top_p": 0.5,
        "stop_sequences": [],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-v2"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("completion")
    return results


# Call Claude model
def call_claude_instant(system_prompt, prompt, bedrock_runtime):
    prompt_config = {
        "prompt": claude_2_prompt_format(system_prompt + prompt),
        "max_tokens_to_sample": 4096,
        "temperature": 0.7,
        "top_k": 250,
        "top_p": 0.5,
        "stop_sequences": [],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-instant-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("completion")
    return results


# Call Cohere model
def call_cohere_command(system_prompt, prompt, bedrock_runtime):
    prompt_config = {
        "prompt": system_prompt + prompt,
        "max_tokens": 2048,
        "temperature": 0.7,
    }

    body = json.dumps(prompt_config)

    modelId = "cohere.command-text-v14"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("generations")[0].get("text")
    return results


def call_cohere_light(system_prompt, prompt, bedrock_runtime):
    prompt_config = {
        "prompt": system_prompt + prompt,
        "max_tokens": 2048,
        "temperature": 0.7,
    }

    body = json.dumps(prompt_config)

    modelId = "cohere.command-light-text-v14"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("generations")[0].get("text")
    return results


# Call Titan model
def call_titan_express(system_prompt, prompt, bedrock_runtime):
    prompt_config = {
        "inputText": system_prompt + prompt,
        "textGenerationConfig": {
            "maxTokenCount": 4096,
            "stopSequences": [],
            "temperature": 0.7,
            "topP": 1,
        },
    }

    body = json.dumps(prompt_config)

    modelId = "amazon.titan-text-express-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("results")[0].get("outputText")
    return results


def call_titan_lite(system_prompt, prompt, bedrock_runtime):
    prompt_config = {
        "inputText": system_prompt + prompt,
        "textGenerationConfig": {
            "maxTokenCount": 4096,
            "stopSequences": [],
            "temperature": 0.7,
            "topP": 1,
        },
    }

    body = json.dumps(prompt_config)

    modelId = "amazon.titan-text-lite-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("results")[0].get("outputText")
    return results


def call_llama2_13b(system_prompt, prompt, bedrock_runtime):
    prompt_config = {
        "prompt": system_prompt + prompt,
        "max_gen_len": 2048,
        "top_p": 0.9,
        "temperature": 0.7,
    }

    body = json.dumps(prompt_config)

    modelId = "meta.llama2-13b-chat-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body["generation"].strip()
    return results


def call_llama2_70b(system_prompt, prompt, bedrock_runtime):
    prompt_config = {
        "prompt": system_prompt + prompt,
        "max_gen_len": 2048,
        "top_p": 0.9,
        "temperature": 0.7,
    }

    body = json.dumps(prompt_config)

    modelId = "meta.llama2-70b-chat-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body["generation"].strip()
    return results


def call_bedrock_model(model: str, system_prompt: str, prompt: str, bedrock_runtime):

    # switch statement for models

    if model == "mistral_large":
        return call_mistral_large(system_prompt, prompt, bedrock_runtime)

    if model == "mistral_8x7b":
        return call_mistral_8x7b(system_prompt, prompt, bedrock_runtime)

    if model == "mistral_7b":
        return call_mistral_7b(system_prompt, prompt, bedrock_runtime)

    if model == "ai21_ultra":
        return call_ai21_ultra(system_prompt, prompt, bedrock_runtime)

    if model == "ai21_mid":
        return call_ai21_mid(system_prompt, prompt, bedrock_runtime)

    if model == "claude_3_opus":
        return call_claude_3_opus(system_prompt, prompt, bedrock_runtime)

    if model == "claude_3_sonnet":
        return call_claude_3_sonnet(system_prompt, prompt, bedrock_runtime)

    if model == "claude_3_haiku":
        return call_claude_3_haiku(system_prompt, prompt, bedrock_runtime)

    if model == "claude_2_1":
        return call_claude_2_1(system_prompt, prompt, bedrock_runtime)

    if model == "claude_2":
        return call_claude_2(system_prompt, prompt, bedrock_runtime)

    if model == "claude_instant":
        return call_claude_instant(system_prompt, prompt, bedrock_runtime)

    if model == "cohere_command":
        return call_cohere_command(system_prompt, prompt, bedrock_runtime)

    if model == "cohere_light":
        return call_cohere_light(system_prompt, prompt, bedrock_runtime)

    if model == "titan_express":
        return call_titan_express(system_prompt, prompt, bedrock_runtime)

    if model == "titan_lite":
        return call_titan_lite(system_prompt, prompt, bedrock_runtime)

    if model == "llama2_13b":
        return call_llama2_13b(system_prompt, prompt, bedrock_runtime)

    if model == "llama2_70b":
        return call_llama2_70b(system_prompt, prompt, bedrock_runtime)

    return None
