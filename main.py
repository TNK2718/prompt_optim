import random
import copy
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- 1. LLM のセットアップ ---
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# --- 2. プロンプト修正用テンプレートとチェーン ---
modify_template = PromptTemplate(
    input_variables=["current_prompt", "seed_instruction"],
    template=(
        "現在のプロンプトは以下です:\n{current_prompt}\n\n"
        "次の指示に従い、プロンプトを改善してください:\n{seed_instruction}\n\n"
        "改善したプロンプトのみ出力してください。"
    )
)


def modify_prompt(current_prompt: str, seed_instruction: str, num_candidates: int = 3) -> list:
    chain = LLMChain(llm=llm, prompt=modify_template)
    candidates = []
    for _ in range(num_candidates):
        candidate = chain.run(current_prompt=current_prompt,
                              seed_instruction=seed_instruction)
        candidates.append(candidate.strip())
    return candidates

# --- 3. ダミーの損失関数 ---


def compute_loss(generator_prompt: dict, discriminator_prompt: dict, samples: list) -> float:
    """
    Generator と Discriminator のプロンプトおよびサンプルに対して、評価指標に基づいた損失を返す（ここではランダム値）。
    実際は生成結果と正解データとの比較など、適切な評価指標を用います。
    """
    return random.random()

# --- 4. アドバーサリアル in-context learning 最適化 ---


def adversarial_incontext_optimization(generator_prompt: dict,
                                       discriminator_prompt: dict,
                                       seed_instruction: str,
                                       seed_demo: str,
                                       samples: list,
                                       T: int = 5,
                                       m: int = 3,
                                       r: int = 2) -> tuple:
    """
    Generator と Discriminator のプロンプト（instruction および demonstrations）を
    交互に最適化するシンプルな実装例。
    """
    for t in range(T):
        sampled = random.sample(samples, m)
        current_loss = compute_loss(
            generator_prompt, discriminator_prompt, sampled)
        print(f"Iteration {t+1}/{T}, Current Loss: {current_loss:.4f}")

        # ◆ Discriminator の instruction 更新（損失を増加させる候補を採用）
        disc_candidates = modify_prompt(
            discriminator_prompt["instruction"], seed_instruction, r)
        best_disc_inst = discriminator_prompt["instruction"]
        best_disc_loss = current_loss
        for cand in disc_candidates:
            temp_disc = copy.deepcopy(discriminator_prompt)
            temp_disc["instruction"] = cand
            loss = compute_loss(generator_prompt, temp_disc, sampled)
            print("【Discriminator candidate instruction】")
            print(cand, f"-> Loss: {loss:.4f}")
            if loss > best_disc_loss:
                best_disc_loss = loss
                best_disc_inst = cand
        discriminator_prompt["instruction"] = best_disc_inst

        # ◆ Discriminator の demonstrations 更新
        new_disc_demos = []
        for demo in discriminator_prompt["demonstrations"]:
            demo_candidates = modify_prompt(demo, seed_demo, r)
            best_demo = demo
            best_demo_loss = current_loss
            for cand in demo_candidates:
                temp_disc = copy.deepcopy(discriminator_prompt)
                idx = discriminator_prompt["demonstrations"].index(demo)
                temp_disc["demonstrations"][idx] = cand
                loss = compute_loss(generator_prompt, temp_disc, sampled)
                print("【Discriminator candidate demo】")
                print(cand, f"-> Loss: {loss:.4f}")
                if loss > best_demo_loss:
                    best_demo_loss = loss
                    best_demo = cand
            new_disc_demos.append(best_demo)
        discriminator_prompt["demonstrations"] = new_disc_demos

        # ◆ Generator の instruction 更新（損失を低下させる候補を採用）
        gen_candidates = modify_prompt(
            generator_prompt["instruction"], seed_instruction, r)
        best_gen_inst = generator_prompt["instruction"]
        best_gen_loss = current_loss
        for cand in gen_candidates:
            temp_gen = copy.deepcopy(generator_prompt)
            temp_gen["instruction"] = cand
            loss = compute_loss(temp_gen, discriminator_prompt, sampled)
            print("【Generator candidate instruction】")
            print(cand, f"-> Loss: {loss:.4f}")
            if loss < best_gen_loss:
                best_gen_loss = loss
                best_gen_inst = cand
        generator_prompt["instruction"] = best_gen_inst

        # ◆ Generator の demonstrations 更新
        new_gen_demos = []
        for demo in generator_prompt["demonstrations"]:
            demo_candidates = modify_prompt(demo, seed_demo, r)
            best_demo = demo
            best_demo_loss = current_loss
            for cand in demo_candidates:
                temp_gen = copy.deepcopy(generator_prompt)
                idx = generator_prompt["demonstrations"].index(demo)
                temp_gen["demonstrations"][idx] = cand
                loss = compute_loss(temp_gen, discriminator_prompt, sampled)
                print("【Generator candidate demo】")
                print(cand, f"-> Loss: {loss:.4f}")
                if loss < best_demo_loss:
                    best_demo_loss = loss
                    best_demo = cand
            new_gen_demos.append(best_demo)
        generator_prompt["demonstrations"] = new_gen_demos

        print("")
    return generator_prompt, discriminator_prompt


# --- 5. メイン処理 ---
if __name__ == "__main__":
    # 初期の Generator プロンプト
    generator_prompt = {
        "instruction": "入力に基づいて創造的な文章を生成してください。",
        "demonstrations": [
            "例1: 昔々、ある村に不思議な出来事が起こりました。",
            "例2: その後、勇敢な主人公が現れ、冒険が始まりました。"
        ]
    }
    # 初期の Discriminator プロンプト
    discriminator_prompt = {
        "instruction": "生成された文章の質と一貫性を評価してください。",
        "demonstrations": [
            "例1: 文章は具体的でわかりやすい。",
            "例2: 説明が不足しており、改善が必要です。"
        ]
    }
    # プロンプト改善用のシード（instruction, demonstration 共通）
    seed_instruction = "このプロンプトを、より難解かつ効果的なものに改善してください。"
    seed_demo = "この例文を、より詳細で明確な表現に書き換えてください。"
    # サンプルデータ（タスクの入力例など）
    samples = ["サンプル入力1", "サンプル入力2", "サンプル入力3", "サンプル入力4", "サンプル入力5"]

    optimized_gen, optimized_disc = adversarial_incontext_optimization(
        generator_prompt,
        discriminator_prompt,
        seed_instruction,
        seed_demo,
        samples,
        T=3, m=2, r=2
    )

    print("=== 最適化後の Generator プロンプト ===")
    print(optimized_gen)
    print("=== 最適化後の Discriminator プロンプト ===")
    print(optimized_disc)
