import argparse
import asyncio
import os
import pickle as pkl
import time

import numpy as np
from tqdm import tqdm

from poke_env import AccountConfiguration, ShowdownServerConfiguration
from poke_env.player import LLMPlayer, SimpleHeuristicsPlayer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--backend",
    type=str,
    default="claude_3_sonnet",
    choices=[
        "claude_3_sonnet",
        "mistral_8x7b",
        "titan_express",
        "claude_3_haiku",
        "mistral_large",
        "claude_3_opus",
    ],
)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--prompt_algo", default="io", choices=["io", "sc", "cot", "tot"])
parser.add_argument("--log_dir", type=str, default="./battle_log/pokellmon_vs_bot")
args = parser.parse_args()


async def main():

    heuristic_player = SimpleHeuristicsPlayer(battle_format="gen8randombattle")

    os.makedirs(args.log_dir, exist_ok=True)
    llm_player = LLMPlayer(
        battle_format="gen8randombattle",
        # api_key="",
        backend=args.backend,
        temperature=args.temperature,
        prompt_algo=args.prompt_algo,
        log_dir=args.log_dir,
        account_configuration=AccountConfiguration(args.backend, ""),
        save_replays=args.log_dir,
    )

    # dynamax is disabled for local battles.
    heuristic_player._dynamax_disable = True
    llm_player._dynamax_disable = True

    if not os.path.exists("baseline.csv"):
        with open("baseline.csv", "w") as f:
            f.write("id,model,temperature,win\n")

    # play against bot for 5 battles
    for i in tqdm(range(5)):
        x = np.random.randint(0, 100)
        if x > 50:
            await heuristic_player.battle_against(llm_player, n_battles=1)
        else:
            await llm_player.battle_against(heuristic_player, n_battles=1)

        # Get the latest battle from llm_player.battles
        latest_battle_id = max(llm_player.battles.keys())
        latest_battle = llm_player.battles[latest_battle_id]

        # Save the latest battle data
        with open(f"{args.log_dir}/{latest_battle_id}.pkl", "wb") as f:
            pkl.dump(latest_battle, f)

        # Write the latest battle result to the CSV file
        with open("baseline.csv", "a") as f:
            f.write(
                f"{latest_battle_id},{args.backend},{args.temperature},"
                + f"{latest_battle.won}\n"
            )


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
