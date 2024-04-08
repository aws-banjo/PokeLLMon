import asyncio
import time
from tqdm import tqdm
import numpy as np
from poke_env import AccountConfiguration, ShowdownServerConfiguration
import os
import pickle as pkl
import argparse
import os
import datetime
import random


from poke_env.player import LLMPlayer


def random_bedrock_model():

    models = [
        "mistral_8x7b",
        "mistral_7b",
        "ai21_ultra",
        "ai21_mid",
        "claude_3_sonnet",
        "claude_3_haiku",
        "claude_3_opus",
        "claude_2",
        "claude_2_1",
        "claude_instant",
        "cohere_command",
        "cohere_light",
        "titan_express",
        "titan_lite",
    ]

    random.seed()
    # Generate a pair of random models
    random_model = random.choice(models)

    return random_model


# TODO replace how you pick models
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
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--prompt_algo", default="io", choices=["io", "sc", "cot", "tot"])
parser.add_argument("--log_dir", type=str, default="./battle_log/llm_vs_llm")
args = parser.parse_args()


async def main():

    # heuristic_player = SimpleHeuristicsPlayer(battle_format="gen8randombattle")

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

    # model_2 = random_bedrock_model()
    model_2 = "claude_3_haiku"  # hard code a model
    model_2_temp = args.temperature

    llm_player_2 = LLMPlayer(
        battle_format="gen8randombattle",
        # api_key="",
        backend=model_2,
        temperature=args.temperature,
        prompt_algo=args.prompt_algo,
        log_dir=args.log_dir,
        account_configuration=AccountConfiguration(model_2 + "_2", ""),
        save_replays=args.log_dir,
    )

    # dynamax is disabled for local battles.
    llm_player_2._dynamax_disable = True
    llm_player._dynamax_disable = True

    if not os.path.exists("results.csv"):
        with open("results.csv", "w") as f:
            f.write(
                "id,player_1_model,player_1_temperature,player_2_model,player_2_temperature,player_1_won\n"
            )

    # play against bot for one battle
    for i in tqdm(range(1)):
        x = np.random.randint(0, 100)
        if x > 50:
            await llm_player_2.battle_against(llm_player, n_battles=1)
        else:
            await llm_player.battle_against(llm_player_2, n_battles=1)


        # Get the latest battle from llm_player.battles
        latest_battle_id = max(llm_player.battles.keys())
        latest_battle = llm_player.battles[latest_battle_id]

        # Save the latest battle data
        with open(f"{args.log_dir}/{latest_battle_id}.pkl", "wb") as f:
            pkl.dump(latest_battle, f)

        # Write the latest battle result to the CSV file
        with open("results.csv", "a") as f:
            f.write(
                f"{latest_battle_id},{args.backend},{args.temperature},"
                + f"{model_2},{model_2_temp},{latest_battle.won}\n"
            )


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
