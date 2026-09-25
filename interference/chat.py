"""
Interactive Chat Interface for Q36 with Interference Search capabilities.
"""
import argparse
import os
import readline
import sys
import time

from interference.backends.q36_backend import (
    Q36Backend,
    list_available_models,
    resolve_model_path,
)
from interference.backends.mock_backend import MockBackend
from interference.agent import run_agent_task
from interference.domains.countdown_domain import CountdownDomain
from interference.core import search


BANNER = r"""
  ___ _____   __   ___       _             __                              
 / _ \___ /  / /_ |_ _| _ _ | |_ ___ _ _ / _|___ _ _ ___ _ _  __ ___ ___ 
| (_) |_ \  | '_ \ | | | ' \|  _/ -_) '_|  _/ -_) '_/ -_) ' \/ _/ -_|_-< 
 \__\___/   |_.__/|___||_||_|\__\___|_| |_| \___|_| \___|_||_\__\___/__/ 
                  Interactive Agent with Multi-Branch Search
"""


class InterferenceChatSession:
    def __init__(
        self,
        backend_choice: str = "q36",
        model_choice: str = None,
        verbose: bool = True,
    ):
        self.backend_choice = backend_choice
        self.verbose = verbose
        if backend_choice == "q36":
            self.backend = Q36Backend(model_path=model_choice)
        else:
            self.backend = MockBackend(lambda p: ["Olá! Sou o agente Q36 com busca por interferência."])
        self.history = []

    def get_current_model_info(self) -> str:
        if self.backend_choice != "q36":
            return "Mock / Simulado"
        path = getattr(self.backend, "model_path", "desconhecido")
        name = os.path.basename(path)
        storage = "SSD NVMe" if "/home/server/models" in path else "HDD Externo"
        return f"{name} ({storage})"

    def list_models(self):
        models = list_available_models()
        if not models:
            print("Nenhum modelo GGUF encontrado nos diretórios padrão.")
            return

        current_path = getattr(self.backend, "model_path", "")
        print("\n\033[1;34m=== Modelos Disponíveis no Sistema ===\033[0m")
        for i, m in enumerate(models, 1):
            is_active = (os.path.abspath(m["path"]) == os.path.abspath(current_path))
            tag = "\033[1;32m* [ATIVO]\033[0m" if is_active else " "
            storage_color = "\033[36m" if "SSD" in m["storage"] else "\033[33m"
            print(f"  {tag} {i:2d}. {storage_color}[{m['storage']}]\033[0m {m['name']} ({m['size_gb']:.1f} GB)")
        print("\nPara trocar o modelo, digite: \033[1m/model <número ou nome>\033[0m (ex: \033[36m/model 1\033[0m ou \033[36m/model bonsai\033[0m)\n")

    def switch_model(self, arg_str: str):
        query = arg_str.strip()
        if not query:
            self.list_models()
            return

        success, res = self.backend.set_model(query)
        if success:
            model_name = os.path.basename(res)
            storage = "SSD NVMe (Rápido)" if "/home/server/models" in res else "HDD Externo"
            print(f"\n✅ \033[1;32mModelo alterado com sucesso!\033[0m")
            print(f"   • Modelo: \033[1m{model_name}\033[0m")
            print(f"   • Local: {storage}")
            print(f"   • Caminho: {res}\n")
        else:
            print(f"\n❌ {res}")
            print("Digite \033[1m/models\033[0m para ver a lista de modelos disponíveis.\n")

    def handle_countdown(self, arg_str: str):
        # Format: 24 98 19 3 -> 361
        try:
            parts = arg_str.split("->")
            if len(parts) != 2:
                print("Uso: /countdown 24 98 19 3 -> 361")
                return
            nums = [int(x.strip()) for x in parts[0].split()]
            target = int(parts[1].strip())
            print(f"\n[Interference Search] Resolvendo Countdown: {nums} -> {target}...")
            domain = CountdownDomain(max_branch=12)
            t0 = time.perf_counter()
            res = search(domain, (nums, target), budget=200, width=6, merge=True, verbose=self.verbose)
            dt = (time.perf_counter() - t0) * 1000

            if res.solved and res.solution:
                for v, e in zip(res.solution.numbers, res.solution.exprs):
                    if v == target:
                        print(f"\n✅ \033[1;32mSolução encontrada em {res.rounds} rodadas ({dt:.2f} ms):\033[0m {e} = {target}")
                        print(f"   • Estados fundidos (Interferência Construtiva): {res.merged_away}")
                        print(f"   • Ramos podados (Interferência Destrutiva): {res.dead_ends}")
            else:
                print("❌ Não foi possível encontrar uma solução exata.")
        except Exception as e:
            print(f"Erro ao processar: {e}")

    def handle_fix(self, arg_str: str):
        # Format: <file> ; <test_cmd> ; <task>
        parts = [p.strip() for p in arg_str.split(";")]
        if len(parts) < 3:
            print("Uso: /fix arquivo.py ; pytest tests/ ; Descrição da tarefa")
            return
        target_file, test_cmd, task = parts[0], parts[1], parts[2]
        if not os.path.exists(target_file):
            print(f"Erro: Arquivo '{target_file}' não encontrado.")
            return

        run_agent_task(
            target_file=target_file,
            test_cmd=test_cmd,
            task_instruction=task,
            backend_choice=self.backend_choice,
            model_choice=getattr(self.backend, "model_path", None),
            verbose=self.verbose,
        )

    def toggle_verbose(self, arg_str: str):
        arg = arg_str.lower().strip()
        if arg in ["on", "1", "true", "ativar"]:
            self.verbose = True
        elif arg in ["off", "0", "false", "desativar"]:
            self.verbose = False
        else:
            self.verbose = not self.verbose
        status = "\033[1;32mATIVADO\033[0m" if self.verbose else "\033[1;31mDESATIVADO\033[0m"
        print(f"[Modo Verboso] {status}")

    def chat_message(self, user_msg: str):
        print(f"\nQ36 ({os.path.basename(self.backend.model_path)}) pensando...", end="", flush=True)
        t0 = time.perf_counter()

        system_prompt = (
            "Você é o assistente de programação Q36 com capacidades de busca por interferência "
            "e raciocínio concorrente. Responda de forma direta, clara e técnica em português."
        )

        full_prompt = f"{system_prompt}\n\nUsuário: {user_msg}\nAssistente:"
        responses, tokens = self.backend.generate_candidates([full_prompt], max_tokens=1024)
        elapsed = time.perf_counter() - t0

        print(f"\r{' ' * 50}\r", end="")  # Clear thinking text
        if responses:
            reply = responses[0].strip()
            print(f"\n{reply}\n")
            speed = tokens / max(elapsed, 0.01)
            print(f"\033[90m[{tokens} tokens gerados em {elapsed:.2f}s (~{speed:.1f} t/s)]\033[0m\n")
        else:
            print("\n[Erro ao obter resposta do modelo]\n")

    def run(self):
        print(BANNER)
        print(f"Modelo Ativo : \033[1;32m{self.get_current_model_info()}\033[0m")
        print(f"Modo Verboso : {'\033[1;32mATIVADO\033[0m' if self.verbose else '\033[1;31mDESATIVADO\033[0m'}")
        print("\nComandos disponíveis:")
        print("  /models                                     -> Lista todos os modelos (SSD e HDD)")
        print("  /model <nome_ou_número>                     -> Troca o modelo ativo em tempo real")
        print("  /fix <arquivo> ; <comando_teste> ; <tarefa> -> Dispara o Interference Search no código")
        print("  /countdown <nums> -> <target>              -> Roda o desafio matemático de Countdown")
        print("  /verbose [on|off]                           -> Ativa/desativa telemetria detalhada de busca")
        print("  /clear                                      -> Limpa o terminal")
        print("  exit ou quit                                -> Sai do chat")
        print("-" * 75)
        print("Ou simplesmente converse normalmente digitando sua mensagem abaixo:\n")

        while True:
            try:
                user_input = input("\033[1;36mq36-interference>\033[0m ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", ":q"]:
                    print("\nEncerrando sessão. Até logo!")
                    break

                if user_input.startswith("/clear"):
                    os.system("clear")
                    continue

                if user_input.startswith("/models"):
                    self.list_models()
                    continue

                if user_input.startswith("/model"):
                    self.switch_model(user_input[len("/model"):].strip())
                    continue

                if user_input.startswith("/verbose"):
                    self.toggle_verbose(user_input[len("/verbose"):].strip())
                    continue

                if user_input.startswith("/countdown"):
                    self.handle_countdown(user_input[len("/countdown"):].strip())
                    continue

                if user_input.startswith("/fix"):
                    self.handle_fix(user_input[len("/fix"):].strip())
                    continue

                # Normal conversational prompt
                self.chat_message(user_input)

            except (KeyboardInterrupt, EOFError):
                print("\nEncerrando sessão.")
                break


def main():
    parser = argparse.ArgumentParser(description="Q36 Interactive Interference Chat")
    parser.add_argument("--backend", choices=["q36", "mock"], default="q36", help="LLM backend")
    parser.add_argument("--model", "-m", default=None, help="Name, number or path of the model to load")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose trace by default")
    args = parser.parse_args()

    session = InterferenceChatSession(
        backend_choice=args.backend,
        model_choice=args.model,
        verbose=not args.quiet,
    )
    session.run()


if __name__ == "__main__":
    main()
