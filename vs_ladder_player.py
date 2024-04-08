import argparse
import asyncio
import os
import pickle as pkl

from tqdm import tqdm

from poke_env import AccountConfiguration, ShowdownServerConfiguration
from poke_env.player import LLMPlayer

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
parser.add_argument(
    "--log_dir", type=str, default="./battle_log/pokellmon_vs_ladder_player"
)
args = parser.parse_args()


async def main():

    os.makedirs(args.log_dir, exist_ok=True)
    llm_player = LLMPlayer(
        battle_format="gen8randombattle",
        #    api_key="Your_openai_api_key",
        backend=args.backend,
        temperature=args.temperature,
        prompt_algo=args.prompt_algo,
        log_dir=args.log_dir,
        account_configuration=AccountConfiguration("Your_account", "Your_password"),
        server_configuration=ShowdownServerConfiguration,
        save_replays=args.log_dir,
    )

    # Playing 5 games on the ladder
    for i in tqdm(range(1)):
        try:
            await llm_player.ladder(1)
            for battle_id, battle in llm_player.battles.items():
                with open(f"{args.log_dir}/{battle_id}.pkl", "wb") as f:
                    pkl.dump(battle, f)
        except:
            continue


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
